"""Utilities for running Spider text-to-SQL experiments with ReMe memory."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SpiderExample:
    """A single Spider example."""

    example_id: str
    db_id: str
    question: str
    gold_sql: str


def load_spider_examples(dataset_path: str | Path) -> list[SpiderExample]:
    """Load Spider examples from a JSON split file."""
    path = Path(dataset_path)
    raw_examples = json.loads(path.read_text(encoding="utf-8"))
    examples: list[SpiderExample] = []
    for index, item in enumerate(raw_examples):
        examples.append(
            SpiderExample(
                example_id=f"{path.stem}:{index}",
                db_id=item["db_id"],
                question=item["question"],
                gold_sql=item["query"],
            ),
        )
    return examples


def build_tables_lookup(tables_json_path: str | Path) -> dict[str, dict[str, Any]]:
    """Build a lookup map for Spider `tables.json`."""
    data = json.loads(Path(tables_json_path).read_text(encoding="utf-8"))
    return {item["db_id"]: item for item in data}


def get_sqlite_path(database_dir: str | Path, db_id: str) -> Path:
    """Resolve the SQLite file for a Spider database."""
    database_dir = Path(database_dir)
    return database_dir / db_id / f"{db_id}.sqlite"


def build_schema_text(
    db_id: str,
    tables_lookup: dict[str, dict[str, Any]],
    sqlite_path: str | Path | None = None,
) -> str:
    """Build a compact schema description from Spider metadata and optional SQLite introspection."""
    if db_id not in tables_lookup:
        raise KeyError(f"db_id '{db_id}' not found in tables.json")

    schema_item = tables_lookup[db_id]
    table_names = schema_item["table_names_original"]
    column_names = schema_item["column_names_original"]
    column_types = schema_item["column_types"]
    primary_keys = set(schema_item["primary_keys"])
    foreign_keys = schema_item["foreign_keys"]

    table_to_columns: dict[int, list[tuple[int, str, str]]] = {i: [] for i in range(len(table_names))}
    for column_index, (table_index, column_name) in enumerate(column_names):
        if table_index == -1:
            continue
        table_to_columns[table_index].append((column_index, column_name, column_types[column_index]))

    lines = [f"Database: {db_id}", "Schema:"]
    for table_index, table_name in enumerate(table_names):
        lines.append(f"- {table_name}")
        for column_index, column_name, column_type in table_to_columns[table_index]:
            tags: list[str] = [column_type]
            if column_index in primary_keys:
                tags.append("PK")
            lines.append(f"  - {column_name} ({', '.join(tags)})")

    if foreign_keys:
        lines.append("Foreign keys:")
        for src_idx, tgt_idx in foreign_keys:
            src_table_idx, src_col = column_names[src_idx]
            tgt_table_idx, tgt_col = column_names[tgt_idx]
            if src_table_idx == -1 or tgt_table_idx == -1:
                continue
            lines.append(
                f"- {table_names[src_table_idx]}.{src_col} -> {table_names[tgt_table_idx]}.{tgt_col}",
            )

    if sqlite_path is not None:
        sqlite_info = build_sqlite_runtime_notes(sqlite_path)
        if sqlite_info:
            lines.append("Runtime notes:")
            lines.extend(f"- {line}" for line in sqlite_info)

    return "\n".join(lines)


def build_sqlite_runtime_notes(sqlite_path: str | Path) -> list[str]:
    """Collect a few runtime notes from SQLite for prompt grounding."""
    path = Path(sqlite_path)
    if not path.exists():
        return [f"SQLite file not found: {path}"]

    notes: list[str] = []
    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()
        table_names = [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
            ).fetchall()
        ]
        notes.append(f"{len(table_names)} tables available in SQLite.")
        for table_name in table_names[:5]:
            try:
                row_count = cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
                notes.append(f'table "{table_name}" has {row_count} rows')
            except sqlite3.Error:
                continue
    finally:
        conn.close()
    return notes


def normalize_sql(sql: str) -> str:
    """Normalize SQL for lightweight exact-match comparison."""
    sql = sql.strip().strip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def extract_sql_from_response(text: str) -> str:
    """Extract plain SQL from an LLM response."""
    text = text.strip()
    fenced = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[0].strip().strip(";")

    sql = text
    if sql.lower().startswith("sql:"):
        sql = sql.split(":", 1)[1].strip()
    return sql.strip().strip(";")


def execute_sql(sqlite_path: str | Path, sql: str) -> tuple[bool, list[tuple[Any, ...]] | str]:
    """Execute SQL and return success plus rows or an error message."""
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        return True, rows
    except sqlite3.Error as err:
        return False, str(err)
    finally:
        conn.close()


def compare_execution_results(
    predicted: list[tuple[Any, ...]] | str,
    gold: list[tuple[Any, ...]] | str,
) -> bool:
    """Compare SQL execution outputs conservatively."""
    if isinstance(predicted, str) or isinstance(gold, str):
        return False
    return predicted == gold


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for a file path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def resolve_jsonl_file_path(path: str | Path) -> Path:
    """Resolve a JSONL file path, tolerating a directory wrapper with a same-named file."""
    resolved = Path(path)
    if resolved.is_dir():
        nested_same_name = resolved / resolved.name
        if nested_same_name.is_file():
            return nested_same_name
        jsonl_files = sorted(p for p in resolved.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl")
        if len(jsonl_files) == 1:
            return jsonl_files[0]
    return resolved


def resolve_memory_store_dir(path: str | Path) -> Path:
    """Resolve a vector-store directory path from a user-supplied memory path."""
    resolved = Path(path)
    if resolved.exists():
        return resolved if resolved.is_dir() else resolved.parent
    if resolved.suffix.lower() == ".jsonl":
        return resolved
    return resolved


def workspace_jsonl_path(store_dir: str | Path, workspace_id: str) -> Path:
    """Return the JSONL file path used by a workspace inside a store directory."""
    return resolve_memory_store_dir(store_dir) / f"{workspace_id}.jsonl"


def dump_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Dump records to a JSONL file."""
    ensure_parent_dir(output_path)
    with Path(output_path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: dict[str, Any], output_path: str | Path) -> None:
    """Append one record to a JSONL file."""
    ensure_parent_dir(output_path)
    with Path(output_path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_jsonl_ids(output_path: str | Path, key: str = "example_id") -> set[str]:
    """Load processed ids from an existing JSONL result file."""
    path = Path(output_path)
    if not path.exists():
        return set()

    processed_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = item.get(key)
            if isinstance(value, str) and value:
                processed_ids.add(value)
    return processed_ids


def load_existing_jsonl_ids_many(output_paths: list[str | Path], key: str = "example_id") -> set[str]:
    """Load processed ids from many JSONL files."""
    processed_ids: set[str] = set()
    for output_path in output_paths:
        processed_ids.update(load_existing_jsonl_ids(output_path, key=key))
    return processed_ids


def merge_jsonl_files_dedup(
    source_paths: list[str | Path],
    target_path: str | Path,
    key: str,
) -> None:
    """Merge JSONL files into one target while deduplicating by a record key."""
    seen: set[str] = set()
    merged_records: list[dict[str, Any]] = []

    for source_path in source_paths:
        path = Path(source_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = record.get(key)
                if not isinstance(value, str) or not value:
                    continue
                if value in seen:
                    continue
                seen.add(value)
                merged_records.append(record)

    dump_jsonl(merged_records, target_path)
