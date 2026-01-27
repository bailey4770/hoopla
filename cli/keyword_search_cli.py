#!/usr/bin/env python3

import argparse
from typing import cast
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    process_string,
    print_search_results,
)


def keyword_search(query: str, limit=DEFAULT_SEARCH_LIMIT) -> list[tuple[int, str]]:
    movies = load_movies()

    search_results: list[tuple[int, str]] = []
    processed_query = process_string(query)

    for movie in movies:
        title: str = movie["title"]
        if processed_query not in process_string(title):
            continue

        id: int = movie["id"]
        search_results.append((id, title))

        if len(search_results) >= limit:
            break

    return search_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    _ = search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    query = cast(str, args.query)

    match cast(str, args.command):
        case "search":
            search_results = keyword_search(query)
            print_search_results(query, search_results)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
