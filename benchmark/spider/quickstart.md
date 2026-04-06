# Spider + ReMe Quick Start

This directory provides a minimal Spider text-to-SQL runner that reuses ReMe task memory.

## What It Does

For each Spider example, the runner can:

1. Retrieve relevant historical task experience from ReMe
2. Generate SQL for the current question
3. Evaluate exact match and lightweight execution match on SQLite
4. Write the finished trajectory back into ReMe as reusable task memory
5. Dump the memory workspace to a JSONL library for reuse across runs

## Dataset Layout

The script expects the standard Spider layout:

```text
spider/
├── train_spider.json
├── dev.json
├── tables.json
└── database/
    ├── academic/
    │   └── academic.sqlite
    └── ...
```

## Environment

You need:

- One OpenAI-compatible LLM endpoint for SQL generation
- One embedding endpoint for ReMe retrieval

The script supports either `LLM_* / EMBEDDING_*` or `FLOW_LLM_* / FLOW_EMBEDDING_*`.

## Build a Memory Library from Train

```bash
python benchmark/spider/run_spider.py \
  --dataset-path /path/to/spider/train_spider.json \
  --tables-json /path/to/spider/tables.json \
  --database-dir /path/to/spider/database \
  --model-name qwen3.5-plus \
  --embedding-model-name text-embedding-v4 \
  --write-memory \
  --resume \
  --stream-output \
  --dump-every 1 \
  --workers 4 \
  --output-file benchmark/spider/exp_result/train_results.jsonl \
  --memory-file benchmark/spider/library/spider_text2sql.jsonl
```

## Evaluate on Dev with Memory Reuse

```bash
python benchmark/spider/run_spider.py \
  --dataset-path /path/to/spider/dev.json \
  --tables-json /path/to/spider/tables.json \
  --database-dir /path/to/spider/database \
  --model-name qwen3.5-plus \
  --embedding-model-name text-embedding-v4 \
  --use-memory \
  --resume \
  --stream-output \
  --write-memory \
  --workers 4 \
  --output-file benchmark/spider/exp_result/dev_results.jsonl \
  --memory-file benchmark/spider/library/spider_text2sql.jsonl
```

## Suggested Workflow

1. Run a small train subset with `--write-memory --limit 100`
2. Re-run dev with `--use-memory`
3. Compare the dev metrics with and without `--use-memory`

## Notes

- Execution match here is a lightweight SQLite result comparison, not the official Spider evaluator.
- The memory library is dumped to JSONL, so you can version it or copy it between runs.
- If you want a strict train/dev separation, build memory only on train, then run dev with `--use-memory` and without `--write-memory`.
- For unstable networks, prefer `--resume`; the runner will automatically enable `--stream-output` and `--dump-every 1` so results and memory are checkpointed after each example.
- Parallel runs use isolated shard workspaces and merge shard outputs at the end, which avoids concurrent writes to the same memory store.
