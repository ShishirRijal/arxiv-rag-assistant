#!/usr/bin/env python3
"""List papers currently ingested in PostgreSQL.

Run inside the app container:

    docker compose exec api python scripts/list_papers.py
    docker compose exec api python scripts/list_papers.py --limit 50
    docker compose exec api python scripts/list_papers.py --parsed parsed
    docker compose exec api python scripts/list_papers.py --category cs.AI
    docker compose exec api python scripts/list_papers.py --search transformer
    docker compose exec api python scripts/list_papers.py --json

The table shows the data available in the `papers` table. Papers with
`pdf_parsed = no` are stored, but are usually not useful for grounded chat/RAG
until parsing and chunk indexing succeed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.database import close_connection_pool, get_db, init_connection_pool


SUMMARY_SQL = """
SELECT
    COUNT(*) AS total_papers,
    COUNT(*) FILTER (WHERE pdf_parsed = TRUE) AS parsed_papers,
    COUNT(*) FILTER (WHERE pdf_parsed = FALSE) AS unparsed_papers,
    MIN(published_at) AS earliest_published_at,
    MAX(published_at) AS latest_published_at
FROM papers;
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List ingested arXiv papers from PostgreSQL.",
        epilog=(
            "Examples:\n"
            "  docker compose exec api python scripts/list_papers.py\n"
            "  docker compose exec api python scripts/list_papers.py --limit 50 --parsed parsed\n"
            "  docker compose exec api python scripts/list_papers.py --category cs.AI\n"
            "  docker compose exec api python scripts/list_papers.py --search transformer\n"
            "  docker compose exec api python scripts/list_papers.py --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=25, help="Rows to display. Default: 25.")
    parser.add_argument("--offset", type=int, default=0, help="Rows to skip. Default: 0.")
    parser.add_argument(
        "--parsed",
        choices=["all", "parsed", "unparsed"],
        default="all",
        help="Filter by PDF parse status. Default: all.",
    )
    parser.add_argument("--category", help="Filter to papers containing this arXiv category.")
    parser.add_argument("--search", help="Case-insensitive search over arxiv_id, title, and abstract.")
    parser.add_argument(
        "--sort",
        choices=["created", "published"],
        default="created",
        help="Sort by ingestion time or publication time. Default: created.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.offset < 0:
        parser.error("--offset must be >= 0")

    return args


def build_query(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    filters = []
    params: dict[str, Any] = {
        "limit": args.limit,
        "offset": args.offset,
    }

    if args.parsed == "parsed":
        filters.append("pdf_parsed = TRUE")
    elif args.parsed == "unparsed":
        filters.append("pdf_parsed = FALSE")

    if args.category:
        filters.append("%(category)s = ANY(categories)")
        params["category"] = args.category

    if args.search:
        filters.append(
            "(arxiv_id ILIKE %(search)s OR title ILIKE %(search)s OR abstract ILIKE %(search)s)"
        )
        params["search"] = f"%{args.search}%"

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    order_by = "published_at DESC NULLS LAST, created_at DESC" if args.sort == "published" else "created_at DESC"

    sql = f"""
    SELECT
        arxiv_id,
        title,
        authors,
        categories,
        published_at,
        pdf_parsed,
        parse_error,
        char_length(full_text) AS full_text_chars,
        created_at,
        updated_at
    FROM papers
    {where_clause}
    ORDER BY {order_by}
    LIMIT %(limit)s OFFSET %(offset)s;
    """
    return sql, params


def fetch_data(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(SUMMARY_SQL)
        summary = dict(cursor.fetchone())

        sql, params = build_query(args)
        cursor.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]

    return summary, rows


def format_date(value: Any) -> str:
    if not value:
        return "-"
    return value.date().isoformat() if hasattr(value, "date") else str(value)


def compact_list(values: Any, *, max_items: int = 2) -> str:
    if not values:
        return "-"
    shown = [str(value) for value in values[:max_items]]
    suffix = f" +{len(values) - max_items}" if len(values) > max_items else ""
    return ", ".join(shown) + suffix


def truncate(value: Any, width: int) -> str:
    text = str(value or "-").replace("\n", " ").strip()
    if len(text) <= width:
        return text
    return text[: max(width - 3, 1)].rstrip() + "..."


def print_table(summary: dict[str, Any], rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    print(
        "Papers: {total} total | {parsed} parsed | {unparsed} unparsed | published {earliest} to {latest}".format(
            total=summary["total_papers"],
            parsed=summary["parsed_papers"],
            unparsed=summary["unparsed_papers"],
            earliest=format_date(summary["earliest_published_at"]),
            latest=format_date(summary["latest_published_at"]),
        )
    )
    print(f"Showing {len(rows)} row(s), offset {args.offset}")
    print()

    columns = [
        ("arxiv_id", "arXiv ID", 12),
        ("published_at", "Published", 10),
        ("pdf_parsed", "Parsed", 6),
        ("categories", "Categories", 18),
        ("authors", "Authors", 28),
        ("full_text_chars", "Chars", 8),
        ("title", "Title", 58),
    ]

    header = "  ".join(label.ljust(width) for _, label, width in columns)
    divider = "  ".join("-" * width for _, _, width in columns)
    print(header)
    print(divider)

    for row in rows:
        display = {
            "arxiv_id": row["arxiv_id"],
            "published_at": format_date(row["published_at"]),
            "pdf_parsed": "yes" if row["pdf_parsed"] else "no",
            "categories": compact_list(row["categories"], max_items=3),
            "authors": compact_list(row["authors"], max_items=2),
            "full_text_chars": row["full_text_chars"] or 0,
            "title": row["title"],
        }
        print("  ".join(truncate(display[key], width).ljust(width) for key, _, width in columns))

    failures = [row for row in rows if not row["pdf_parsed"] and row.get("parse_error")]
    if failures:
        print()
        print("Parse errors in displayed rows:")
        for row in failures:
            print(f"- {row['arxiv_id']}: {truncate(row['parse_error'], 120)}")


def json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def main() -> int:
    args = parse_args()
    init_connection_pool()
    try:
        summary, rows = fetch_data(args)
    finally:
        close_connection_pool()

    if args.json:
        print(json.dumps({"summary": summary, "papers": rows}, indent=2, default=json_default))
    else:
        print_table(summary, rows, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
