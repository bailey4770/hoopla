#!/usr/bin/env python3

import argparse
import sys
from typing import cast

from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    process_string,
    print_search_results,
)

from lib.inverted_index import InvertedIndex


def cmd_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Error: index files not found")
        sys.exit(1)
    except Exception as e:
        print("Error: ", e)
        sys.exit(1)

    processed_query = process_string()(query)

    search_results: list[tuple[int, str]] = []
    for q in processed_query:
        search_results.extend(index.get_document(q))

        if len(search_results) >= limit:
            search_results = search_results[:limit]
            break

    print_search_results(query, search_results)


def cmd_build() -> None:
    movies = load_movies()

    inv_idx = InvertedIndex()
    inv_idx.build(movies)
    inv_idx.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    _ = search_parser.add_argument("query", type=str, help="Search query")

    _ = subparsers.add_parser(
        "build", help="Build inverted index of movie titles and descriptions"
    )

    args = parser.parse_args()

    match cast(str, args.command):
        case "build":
            cmd_build()

        case "search":
            cmd_search(cast(str, args.query))

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
