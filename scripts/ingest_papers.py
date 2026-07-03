#!/usr/bin/env python3
"""Ad-hoc arXiv ingestion helper.

Run inside the app container:

    docker compose exec api python scripts/ingest_papers.py --id 1706.03762
    docker compose exec api python scripts/ingest_papers.py --id https://arxiv.org/abs/1706.03762
    docker compose exec api python scripts/ingest_papers.py --id 1706.03762 --id 2005.11401
    docker compose exec api python scripts/ingest_papers.py --date 2026-07-02 --category cs.AI --max-results 10
    docker compose exec api python scripts/ingest_papers.py --query 'ti:"retrieval augmented generation" AND cat:cs.AI'

The script stores papers in PostgreSQL, syncs the paper OpenSearch index, and
indexes parsed paper chunks for RAG retrieval.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path
from typing import Iterable

import arxiv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.database import close_connection_pool, get_db, init_connection_pool
from src.services.arxiv.client import ArxivClient, RATE_LIMIT_SECONDS
from src.services.embeddings.factory import make_embeddings_service
from src.services.metadata_fetcher import MetadataFetcher
from src.services.opensearch.chunk_indexer import ChunkIndexer
from src.services.opensearch.factory import make_opensearch_client, make_paper_indexer
from src.services.pdf_parser.downloader import PDFDownloader
from src.services.pdf_parser.parser import DoclingParser
from src.services.schemas import ArxivPaper

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest selected arXiv papers into Postgres and OpenSearch.",
        epilog=(
            "Examples:\n"
            "  docker compose exec api python scripts/ingest_papers.py --id 1706.03762\n"
            "  docker compose exec api python scripts/ingest_papers.py --id https://arxiv.org/abs/1706.03762\n"
            "  docker compose exec api python scripts/ingest_papers.py --date 2026-07-02 --category cs.AI --max-results 20\n"
            "  docker compose exec api python scripts/ingest_papers.py --query 'ti:\"RAG\" AND cat:cs.AI' --max-results 10"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--id",
        dest="ids",
        action="append",
        help="arXiv ID or URL. Can be passed more than once.",
    )
    source.add_argument(
        "--query",
        help='Raw arXiv query, for example: ti:"RAG" AND cat:cs.AI',
    )
    source.add_argument(
        "--date",
        type=date.fromisoformat,
        help="Submission date to ingest, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--category",
        help="arXiv category for --date, for example cs.AI, cs.LG, cs.CL.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum results for --query or --date. Default: 20.",
    )
    parser.add_argument(
        "--skip-paper-index",
        action="store_true",
        help="Skip paper-level OpenSearch indexing.",
    )
    parser.add_argument(
        "--skip-chunk-index",
        action="store_true",
        help="Skip chunk-level OpenSearch indexing used by chat/RAG retrieval.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO.",
    )
    args = parser.parse_args()

    if args.date and not args.category:
        parser.error("--date requires --category")
    if args.max_results < 1:
        parser.error("--max-results must be >= 1")

    return args


def normalize_arxiv_id(value: str) -> str:
    """Accept raw IDs and common arXiv abs/pdf URLs."""
    value = value.strip()
    match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", value)
    if match:
        return match.group(1)

    # Older identifiers such as cs/9901001 are uncommon here but valid.
    match = re.search(r"([a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", value)
    if match:
        return match.group(1)

    raise ValueError(f"Could not parse arXiv ID from: {value}")


def result_to_paper(result: arxiv.Result) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=result.entry_id.split("/")[-1],
        title=result.title,
        abstract=result.summary,
        authors=[author.name for author in result.authors],
        categories=result.categories,
        pdf_url=result.pdf_url,
        published_at=result.published,
    )


def fetch_by_ids(ids: Iterable[str]) -> list[ArxivPaper]:
    normalized_ids = [normalize_arxiv_id(value) for value in ids]
    logger.info("Fetching arXiv IDs: %s", ", ".join(normalized_ids))

    client = arxiv.Client(
        page_size=max(len(normalized_ids), 1),
        delay_seconds=RATE_LIMIT_SECONDS,
        num_retries=3,
    )
    search = arxiv.Search(
        id_list=normalized_ids,
        max_results=len(normalized_ids),
    )

    papers: list[ArxivPaper] = []
    for result in client.results(search):
        papers.append(result_to_paper(result))
    return papers


def fetch_papers(args: argparse.Namespace) -> list[ArxivPaper]:
    if args.ids:
        return fetch_by_ids(args.ids)

    client = ArxivClient()
    if args.query:
        return client.fetch_by_query(args.query, max_results=args.max_results)

    return client.fetch_by_date(
        category=args.category,
        target_date=args.date,
        max_results=args.max_results,
    )


def make_fetcher() -> MetadataFetcher:
    return MetadataFetcher(
        arxiv_client=ArxivClient(),
        pdf_downloader=PDFDownloader(),
        pdf_parser=DoclingParser(),
        db=get_db,
    )


def process_papers(fetcher: MetadataFetcher, papers: list[ArxivPaper]) -> dict:
    summary = {
        "total": len(papers),
        "saved": 0,
        "parsed": 0,
        "failed": 0,
        "saved_ids": [],
        "failed_ids": [],
    }

    for paper in papers:
        # MetadataFetcher does not currently expose a public method for an
        # already-fetched paper list, so this operational script reuses the
        # same isolated per-paper pipeline used by the public batch methods.
        result = fetcher._process_one(paper)
        if result.success:
            summary["saved"] += 1
            summary["saved_ids"].append(result.arxiv_id)
            if result.pdf_parsed:
                summary["parsed"] += 1
        else:
            summary["failed"] += 1
            summary["failed_ids"].append(result.arxiv_id)

    return summary


def index_papers(arxiv_ids: list[str], *, paper_index: bool, chunk_index: bool) -> dict:
    summary: dict = {
        "paper_indexed": 0,
        "paper_index_failed": [],
        "chunks_indexed": 0,
        "chunk_index_errors": 0,
        "chunk_index_failed": [],
    }

    if paper_index:
        paper_indexer = make_paper_indexer()
        for arxiv_id in arxiv_ids:
            if paper_indexer.index_one(arxiv_id):
                summary["paper_indexed"] += 1
            else:
                summary["paper_index_failed"].append(arxiv_id)

    if chunk_index:
        chunk_indexer = ChunkIndexer(
            os_client=make_opensearch_client(),
            embeddings_svc=make_embeddings_service(),
            db=get_db,
        )
        chunk_indexer.setup()

        for arxiv_id in arxiv_ids:
            result = chunk_indexer.index_paper(arxiv_id)
            summary["chunks_indexed"] += result.indexed
            summary["chunk_index_errors"] += result.errors
            if result.errors:
                summary["chunk_index_failed"].append(arxiv_id)

    return summary


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    init_connection_pool()
    try:
        papers = fetch_papers(args)
        if not papers:
            print(json.dumps({"status": "no_papers_found"}, indent=2))
            return 0

        ingest_summary = process_papers(make_fetcher(), papers)
        index_summary = index_papers(
            ingest_summary["saved_ids"],
            paper_index=not args.skip_paper_index,
            chunk_index=not args.skip_chunk_index,
        )

        print(json.dumps({
            "status": "complete",
            "ingestion": ingest_summary,
            "indexing": index_summary,
        }, indent=2))
        return 1 if ingest_summary["failed"] else 0
    finally:
        close_connection_pool()


if __name__ == "__main__":
    raise SystemExit(main())
