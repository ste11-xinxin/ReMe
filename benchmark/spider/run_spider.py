"""Run Spider text-to-SQL with ReMe task-memory reuse."""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from reme_ai import ReMeApp

from spider_utils import (
    append_jsonl,
    SpiderExample,
    build_schema_text,
    build_tables_lookup,
    compare_execution_results,
    dump_jsonl,
    ensure_parent_dir,
    execute_sql,
    extract_sql_from_response,
    get_sqlite_path,
    load_existing_jsonl_ids,
    load_existing_jsonl_ids_many,
    load_spider_examples,
    merge_jsonl_files_dedup,
    normalize_sql,
    resolve_jsonl_file_path,
    resolve_memory_store_dir,
    workspace_jsonl_path,
)

""".\.venv\Scripts\python.exe .\benchmark\spider\run_spider.py `
  --dataset-path D:\ReMe\benchmark\spider\spider_data\spider_data\train_spider.json `
  --tables-json D:\ReMe\benchmark\spider\spider_data\spider_data\tables.json `
  --database-dir D:\ReMe\benchmark\spider\spider_data\spider_data\database `
  --model-name qwen3.5-plus `
  --embedding-model-name text-embedding-v4 `
  --write-memory `
  --resume `
  --stream-output `
  --dump-every 1 `
  --offset 0 `
  --output-file D:\ReMe\benchmark\spider\exp_result\train_memory_build.jsonl `
  --memory-file D:\ReMe\benchmark\spider\library\spider_text2sql.jsonl
"""


SYSTEM_PROMPT = """You are an expert text-to-SQL assistant for the Spider benchmark.
You must write one SQLite-compatible SQL query that answers the question.
Return SQL only. Do not include explanations, markdown fences, or commentary."""


@dataclass(slots=True)
class SpiderResult:
    """Evaluation result for one Spider example."""

    example_id: str
    db_id: str
    question: str
    gold_sql: str
    predicted_sql: str
    exact_match: bool
    execution_match: bool
    prediction_success: bool
    prediction_error: str | None
    retrieved_experience: str


def resolve_env(primary_key: str, secondary_key: str) -> str | None:
    """Resolve env values with a fallback key."""
    return os.getenv(primary_key) or os.getenv(secondary_key) or None


class SpiderMemoryRunner:
    """Spider runner that reuses ReMe task memory."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.tables_lookup = build_tables_lookup(args.tables_json)
        self.processed_ids = load_existing_jsonl_ids(args.output_file) if args.resume else set()
        self.pending_memory_records: list[dict[str, Any]] = []
        self.sql_client = AsyncOpenAI(
            api_key=resolve_env("LLM_API_KEY", "FLOW_LLM_API_KEY") or args.llm_api_key,
            base_url=resolve_env("LLM_BASE_URL", "FLOW_LLM_BASE_URL") or args.llm_base_url,
        )

    async def generate_sql(
        self,
        example: SpiderExample,
        schema_text: str,
        retrieved_experience: str,
    ) -> str:
        """Call the SQL generation model."""
        user_prompt = (
            f"Question:\n{example.question}\n\n"
            f"{schema_text}\n\n"
            "Relevant prior experience:\n"
            f"{retrieved_experience or 'No retrieved experience.'}\n\n"
            "Write the final SQLite query."
        )

        response = await self.sql_client.chat.completions.create(
            model=self.args.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.args.temperature,
        )
        content = response.choices[0].message.content or ""
        return extract_sql_from_response(content)

    async def retrieve_experience(
        self,
        app: ReMeApp | None,
        example: SpiderExample,
        schema_text: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Retrieve task memories relevant to the current question."""
        if app is None or not self.args.use_memory:
            return "", []

        query = (
            "Spider text-to-sql task\n"
            f"Database: {example.db_id}\n"
            f"Question: {example.question}\n"
            f"{schema_text}"
        )
        result = await app.async_execute(
            name="retrieve_task_memory",
            workspace_id=self.args.workspace_id,
            query=query,
            top_k=self.args.top_k,
            enable_llm_rerank=self.args.enable_llm_rerank,
            enable_score_filter=self.args.enable_score_filter,
            enable_llm_rewrite=self.args.enable_llm_rewrite,
        )
        metadata = result.get("metadata", {}) or {}
        return result.get("answer", ""), metadata.get("memory_list", []) or []

    async def record_memory_usage(
        self,
        app: ReMeApp | None,
        memory_dicts: list[dict[str, Any]],
        update_utility: bool,
    ) -> None:
        """Update retrieval frequency and optional utility for recalled memories."""
        if app is None or not memory_dicts:
            return
        await app.async_execute(
            name="record_task_memory",
            workspace_id=self.args.workspace_id,
            memory_dicts=memory_dicts,
            update_utility=update_utility,
        )

    async def summarize_trajectory(
        self,
        app: ReMeApp | None,
        example: SpiderExample,
        schema_text: str,
        predicted_sql: str,
        exact_match: bool,
        execution_match: bool,
        prediction_error: str | None,
    ) -> None:
        """Summarize one trajectory into task memory."""
        if app is None or not self.args.write_memory:
            return

        score = 1.0 if execution_match else 0.0
        feedback = (
            f"Gold SQL: {example.gold_sql}\n"
            f"Predicted SQL: {predicted_sql}\n"
            f"Exact match: {exact_match}\n"
            f"Execution match: {execution_match}\n"
            f"Error: {prediction_error or 'None'}"
        )
        trajectories = [
            {
                "task_id": example.example_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Spider text-to-sql task\nQuestion: {example.question}\n\n"
                            f"{schema_text}"
                        ),
                    },
                    {"role": "assistant", "content": predicted_sql},
                    {"role": "user", "content": feedback},
                ],
                "score": score,
            },
        ]
        result = await app.async_execute(
            name="summary_task_memory",
            workspace_id=self.args.workspace_id,
            trajectories=trajectories,
            success_threshold=1.0,
            enable_soft_comparison=True,
            validation_threshold=0.5,
        )
        memory_list = (result.get("metadata", {}) or {}).get("memory_list", []) or []
        for memory in memory_list:
            if hasattr(memory, "model_dump"):
                memory = memory.model_dump()
            if not isinstance(memory, dict):
                continue
            memory_id = memory.get("memory_id")
            if not isinstance(memory_id, str) or not memory_id:
                continue
            self.pending_memory_records.append(memory)

    async def maybe_load_memory_file(self, app: ReMeApp | None) -> None:
        """Load a dumped memory library if present."""
        if app is None or not self.args.memory_file:
            return

        memory_dir = resolve_memory_store_dir(self.args.load_memory_file or self.args.memory_file)
        memory_path = workspace_jsonl_path(memory_dir, self.args.workspace_id)
        if not memory_path.exists():
            logger.info("Memory file does not exist yet, starting with an empty workspace.")
            return

        await app.async_execute(
            name="vector_store",
            workspace_id=self.args.workspace_id,
            action="load",
            path=str(memory_dir),
        )
        logger.info(f"Loaded memory library from {memory_path}")

    async def maybe_dump_memory_file(self, app: ReMeApp | None) -> None:
        """Dump the current workspace memory library to disk."""
        if app is None or not self.args.memory_file:
            return

        memory_dir = resolve_memory_store_dir(self.args.memory_file)
        ensure_parent_dir(memory_dir)
        Path(memory_dir).mkdir(parents=True, exist_ok=True)
        if getattr(self.args, "dump_memory_delta_only", False):
            memory_path = workspace_jsonl_path(memory_dir, self.args.workspace_id)
            existing_ids = load_existing_jsonl_ids(memory_path, key="memory_id")
            for memory in self.pending_memory_records:
                memory_id = memory.get("memory_id")
                if not isinstance(memory_id, str) or not memory_id or memory_id in existing_ids:
                    continue
                append_jsonl(memory, memory_path)
                existing_ids.add(memory_id)
            logger.info(f"Dumped memory delta to {memory_path}")
            return
        await app.async_execute(
            name="vector_store",
            workspace_id=self.args.workspace_id,
            action="dump",
            path=str(memory_dir),
        )
        logger.info(f"Dumped memory library to {workspace_jsonl_path(memory_dir, self.args.workspace_id)}")

    def append_result(self, result: SpiderResult) -> None:
        """Append a finished result to disk immediately."""
        append_jsonl(asdict(result), self.args.output_file)
        self.processed_ids.add(result.example_id)

    async def evaluate_example(self, app: ReMeApp | None, example: SpiderExample) -> SpiderResult:
        """Run one Spider example end to end."""
        sqlite_path = get_sqlite_path(self.args.database_dir, example.db_id)
        schema_text = build_schema_text(example.db_id, self.tables_lookup, sqlite_path=sqlite_path)
        retrieved_experience, memory_dicts = await self.retrieve_experience(app, example, schema_text)
        predicted_sql = await self.generate_sql(example, schema_text, retrieved_experience)

        prediction_success, predicted_output = execute_sql(sqlite_path, predicted_sql)
        _, gold_output = execute_sql(sqlite_path, example.gold_sql)
        exact_match = normalize_sql(predicted_sql) == normalize_sql(example.gold_sql)
        execution_match = prediction_success and compare_execution_results(predicted_output, gold_output)
        prediction_error = None if prediction_success else str(predicted_output)

        await self.record_memory_usage(app, memory_dicts, update_utility=execution_match)
        await self.summarize_trajectory(
            app=app,
            example=example,
            schema_text=schema_text,
            predicted_sql=predicted_sql,
            exact_match=exact_match,
            execution_match=execution_match,
            prediction_error=prediction_error,
        )

        return SpiderResult(
            example_id=example.example_id,
            db_id=example.db_id,
            question=example.question,
            gold_sql=example.gold_sql,
            predicted_sql=predicted_sql,
            exact_match=exact_match,
            execution_match=execution_match,
            prediction_success=prediction_success,
            prediction_error=prediction_error,
            retrieved_experience=retrieved_experience,
        )

    async def run(self) -> list[SpiderResult]:
        """Run the configured Spider split."""
        examples = getattr(self.args, "examples_override", None) or load_spider_examples(self.args.dataset_path)
        if getattr(self.args, "examples_override", None) is None:
            examples = examples[self.args.offset :]
            if self.args.limit is not None:
                examples = examples[: self.args.limit]
        if self.processed_ids:
            examples = [example for example in examples if example.example_id not in self.processed_ids]

        reme_app: ReMeApp | None = None
        if self.args.use_memory or self.args.write_memory or self.args.memory_file:
            reme_app = ReMeApp(
                f"llm.default.model_name={self.args.memory_model_name or self.args.model_name}",
                f"embedding_model.default.model_name={self.args.embedding_model_name}",
                "vector_store.default.backend=memory",
                llm_api_key=resolve_env("FLOW_LLM_API_KEY", "LLM_API_KEY") or self.args.llm_api_key,
                llm_api_base=resolve_env("FLOW_LLM_BASE_URL", "LLM_BASE_URL") or self.args.llm_base_url,
                embedding_api_key=resolve_env("FLOW_EMBEDDING_API_KEY", "EMBEDDING_API_KEY")
                or self.args.embedding_api_key,
                embedding_api_base=resolve_env("FLOW_EMBEDDING_BASE_URL", "EMBEDDING_BASE_URL")
                or self.args.embedding_base_url,
            )

        results: list[SpiderResult] = []
        if reme_app is None:
            for idx, example in enumerate(examples, start=1):
                logger.info(f"[{idx}/{len(examples)}] Running example={example.example_id} db={example.db_id}")
                result = await self.evaluate_example(None, example)
                results.append(result)
                if self.args.stream_output:
                    self.append_result(result)
            return results

        async with reme_app as app:
            await self.maybe_load_memory_file(app)
            for idx, example in enumerate(examples, start=1):
                logger.info(f"[{idx}/{len(examples)}] Running example={example.example_id} db={example.db_id}")
                result = await self.evaluate_example(app, example)
                results.append(result)
                if self.args.stream_output:
                    self.append_result(result)
                if self.args.dump_every > 0 and idx % self.args.dump_every == 0:
                    await self.maybe_dump_memory_file(app)
                logger.info(
                    "exact_match={} execution_match={} prediction_success={}",
                    result.exact_match,
                    result.execution_match,
                    result.prediction_success,
                )
            await self.maybe_dump_memory_file(app)

        return results

    async def close(self) -> None:
        """Close the SQL client."""
        await self.sql_client.close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Spider text-to-SQL with ReMe experience reuse.")
    parser.add_argument("--dataset-path", required=True, help="Path to Spider split JSON, e.g. dev.json")
    parser.add_argument("--tables-json", required=True, help="Path to Spider tables.json")
    parser.add_argument("--database-dir", required=True, help="Path to Spider database/ directory")
    parser.add_argument("--model-name", default="qwen3.5-plus", help="SQL generation model name")
    parser.add_argument("--memory-model-name", default=None, help="Optional model used by ReMe memory flows")
    parser.add_argument("--embedding-model-name", default="text-embedding-v4", help="Embedding model for memory")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--workspace-id", default="spider_text2sql", help="ReMe workspace id")
    parser.add_argument("--memory-file", default="benchmark/spider/library/spider_text2sql.jsonl")
    parser.add_argument(
        "--load-memory-file",
        default=None,
        help="Optional memory library to load before running; defaults to --memory-file",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of memories to retrieve")
    parser.add_argument(
        "--enable-llm-rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LLM reranking for retrieved memories",
    )
    parser.add_argument(
        "--enable-llm-rewrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LLM rewriting for retrieved memories before prompt injection",
    )
    parser.add_argument(
        "--enable-score-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable score-based filtering for retrieved memories",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of examples to run")
    parser.add_argument("--offset", type=int, default=0, help="Optional offset into the split")
    parser.add_argument("--output-file", default="benchmark/spider/exp_result/results.jsonl")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers. Each worker uses an isolated memory shard.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoints by skipping finished example_id values",
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Append each finished result immediately instead of only writing at the end",
    )
    parser.add_argument(
        "--dump-every",
        type=int,
        default=1,
        help="Dump the memory workspace every N processed examples when memory is enabled",
    )
    parser.add_argument("--use-memory", action="store_true", help="Retrieve prior task memories before generation")
    parser.add_argument("--write-memory", action="store_true", help="Write each finished trajectory into task memory")
    parser.add_argument("--llm-api-key", default=None, help="Override SQL/memory LLM API key")
    parser.add_argument("--llm-base-url", default=None, help="Override SQL/memory LLM base URL")
    parser.add_argument("--embedding-api-key", default=None, help="Override embedding API key")
    parser.add_argument("--embedding-base-url", default=None, help="Override embedding API base URL")
    return parser.parse_args()


def normalize_checkpoint_settings(args: argparse.Namespace) -> None:
    """Make resume mode durable enough for interrupted runs."""
    if not args.resume:
        return

    if not args.stream_output:
        logger.warning("`--resume` requires incremental result checkpoints, enabling `--stream-output`.")
        args.stream_output = True

    if args.dump_every != 1:
        logger.warning("`--resume` works best with per-example memory checkpoints, forcing `--dump-every 1`.")
        args.dump_every = 1


def partition_examples(examples: list[SpiderExample], workers: int) -> list[list[SpiderExample]]:
    """Split examples into balanced shards."""
    if workers <= 1:
        return [examples]
    shards: list[list[SpiderExample]] = [[] for _ in range(workers)]
    for idx, example in enumerate(examples):
        shards[idx % workers].append(example)
    return shards


def shard_path(base_path: str | Path, shard_index: int) -> Path:
    """Build a shard-specific path next to the base path."""
    base = Path(base_path)
    return base.with_name(f"{base.stem}.part{shard_index}{base.suffix}")


def merge_text_files(source_paths: list[Path], target_path: Path) -> None:
    """Concatenate shard text files into a single target file."""
    ensure_parent_dir(target_path)
    with target_path.open("w", encoding="utf-8") as out_file:
        for source_path in source_paths:
            if not source_path.exists():
                continue
            content = source_path.read_text(encoding="utf-8")
            if content and not content.endswith("\n"):
                content += "\n"
            out_file.write(content)


async def run_single_worker(args: argparse.Namespace, examples: list[SpiderExample], worker_index: int = 0) -> list[SpiderResult]:
    """Run one isolated worker shard."""
    worker_args = argparse.Namespace(**vars(args))
    worker_args.examples_override = examples
    if getattr(args, "shared_memory_rounds", False):
        worker_args.workspace_id = args.workspace_id
    else:
        worker_args.workspace_id = f"{args.workspace_id}_w{worker_index}"
    if args.workers > 1:
        worker_output_path = shard_path(args.output_file, worker_index)
        worker_memory_dir = shard_path(resolve_memory_store_dir(args.memory_file), worker_index)
        worker_args.output_file = str(worker_output_path)
        worker_args.memory_file = str(worker_memory_dir)
        if getattr(args, "shared_memory_rounds", False):
            if args.load_memory_file:
                worker_args.load_memory_file = str(resolve_memory_store_dir(args.load_memory_file))
            elif args.resume:
                global_memory_dir = resolve_memory_store_dir(args.memory_file)
                global_memory_path = workspace_jsonl_path(global_memory_dir, worker_args.workspace_id)
                if global_memory_path.exists():
                    worker_args.load_memory_file = str(global_memory_dir)
            worker_args.dump_memory_delta_only = True
        else:
            worker_memory_path = workspace_jsonl_path(worker_memory_dir, worker_args.workspace_id)

            shard_load_dir: Path | None = None
            if args.load_memory_file:
                candidate = shard_path(resolve_memory_store_dir(args.load_memory_file), worker_index)
                candidate_path = workspace_jsonl_path(candidate, worker_args.workspace_id)
                if candidate_path.exists():
                    shard_load_dir = candidate

            if shard_load_dir is not None:
                worker_args.load_memory_file = str(shard_load_dir)
            elif args.resume and worker_memory_path.exists():
                worker_args.load_memory_file = str(worker_memory_dir)
            elif args.load_memory_file:
                worker_args.load_memory_file = str(resolve_memory_store_dir(args.load_memory_file))
            else:
                worker_args.load_memory_file = str(resolve_memory_store_dir(args.memory_file))
    runner = SpiderMemoryRunner(worker_args)
    try:
        return await runner.run()
    finally:
        await runner.close()


async def async_main() -> None:
    """Async entry point."""
    args = parse_args()
    normalize_checkpoint_settings(args)
    if args.workers <= 1:
        runner = SpiderMemoryRunner(args)
        try:
            results = await runner.run()
        finally:
            await runner.close()

        if not args.stream_output:
            output_records = [asdict(result) for result in results]
            dump_jsonl(output_records, args.output_file)
    else:
        all_examples = load_spider_examples(args.dataset_path)
        all_examples = all_examples[args.offset :]
        if args.limit is not None:
            all_examples = all_examples[: args.limit]

        shard_output_paths = [shard_path(args.output_file, i) for i in range(args.workers)]
        memory_store_dir = resolve_memory_store_dir(args.memory_file)
        memory_file_path = workspace_jsonl_path(memory_store_dir, args.workspace_id)
        resume_output_sources = [args.output_file, *shard_output_paths] if args.resume else [args.output_file]
        processed_ids = load_existing_jsonl_ids_many(resume_output_sources)
        if processed_ids:
            all_examples = [example for example in all_examples if example.example_id not in processed_ids]
        round_examples = [all_examples[i : i + args.workers] for i in range(0, len(all_examples), args.workers)]
        shard_memory_paths = [workspace_jsonl_path(shard_path(memory_store_dir, i), args.workspace_id) for i in range(args.workers)]

        logger.info(
            "Running {} examples across {} workers in {} synchronized round(s).",
            len(all_examples),
            args.workers,
            len(round_examples),
        )

        results = []
        for round_idx, current_round in enumerate(round_examples, start=1):
            round_args = argparse.Namespace(**vars(args))
            round_args.shared_memory_rounds = True
            round_args.dump_memory_delta_only = True
            if memory_file_path.exists():
                round_args.load_memory_file = str(memory_store_dir)
            elif args.load_memory_file:
                round_args.load_memory_file = args.load_memory_file

            logger.info(
                "Round {}/{} starting with {} example(s).",
                round_idx,
                len(round_examples),
                len(current_round),
            )

            tasks = [
                run_single_worker(round_args, [example], worker_index=i)
                for i, example in enumerate(current_round)
            ]
            results_by_shard = await asyncio.gather(*tasks)
            results.extend(result for shard_results in results_by_shard for result in shard_results)

            merge_jsonl_files_dedup([args.output_file, *shard_output_paths], Path(args.output_file), key="example_id")
            merge_jsonl_files_dedup([memory_file_path, *shard_memory_paths], memory_file_path, key="memory_id")

    total = len(results)
    exact = sum(result.exact_match for result in results)
    execution = sum(result.execution_match for result in results)
    print(f"Saved {total} results to {args.output_file}")
    if total:
        print(f"Exact match: {exact}/{total} = {exact / total:.2%}")
        print(f"Execution match: {execution}/{total} = {execution / total:.2%}")


def main() -> None:
    """Sync entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
