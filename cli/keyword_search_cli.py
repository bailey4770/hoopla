#!/usr/bin/env python3

import argparse
import json

from typing import TypedDict, cast

MOVIES_FILE_PATH = "./data/movies.json"


class Movie(TypedDict):
    id: int
    title: str
    description: str


Movies = list[Movie]


class MovieData(TypedDict):
    movies: Movies


def keyword_search(query: str) -> list[tuple[int, str]]:
    with open(MOVIES_FILE_PATH, "r") as f:
        movies_data = cast(MovieData, json.load(f))

    search_results: list[tuple[int, str]] = []

    for movie in movies_data["movies"]:
        title: str = movie["title"]

        if query in title:
            id: int = movie["id"]
            search_results.append((id, title))

    return search_results


def print_search_results(query: str, search_results: list[tuple[int, str]]):
    ordered = sorted(search_results, key=lambda movie: movie[0])
    truncated = ordered[:5]

    print(f"Searching for: {query}")

    count = 0
    for res in truncated:
        count += 1
        print(f"{count}. {res[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    _ = search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            search_results = keyword_search(args.query)
            print_search_results(args.query, search_results)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
