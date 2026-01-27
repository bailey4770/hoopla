#!/usr/bin/env python3

import argparse
from typing import cast

from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    process_string,
    print_search_results,
)

from lib.inverted_index import InvertedIndex


def keyword_search(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[tuple[int, str]]:
    movies = load_movies()
    processor = process_string()

    search_results: list[tuple[int, str]] = []
    processed_query = processor(query)

    for movie in movies:
        title = movie["title"]
        processed_title = processor(title)

        if any(q in t for q in processed_query for t in processed_title):
            search_results.append((movie["id"], title))

            if len(search_results) == limit:
                break

    return search_results


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
            movies = load_movies()

            inv_idx = InvertedIndex()
            inv_idx.build(movies)
            inv_idx.save()

            # hardcoded test
            term = "merida"
            docs = inv_idx.get_document(term)
            print(f"First document for token 'merida' = {docs[0]}")

        case "search":
            query = cast(str, args.query)
            search_results = keyword_search(query)
            print_search_results(query, search_results)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
