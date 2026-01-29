#!/usr/bin/env python3

import argparse
from typing import cast

from lib.semantic_search import SemanticSearch
from lib.search_utils import load_movies, DEFAULT_SEARCH_LIMIT


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser("verify", help="Verify loaded model")

    embded_text_parser = subparsers.add_parser(
        "embed_text", help="Embdeds inputted text to semantic vector"
    )
    _ = embded_text_parser.add_argument("text", type=str, help="Text to be embedded")

    _ = subparsers.add_parser(
        "verify_embeddings",
        help="Verifies the embeddings file and creates it if one does not exist",
    )

    embded_query_parser = subparsers.add_parser(
        "embedquery", help="Embdeds query to semantic vector"
    )
    _ = embded_query_parser.add_argument("query", type=str, help="Query to be embedded")

    search_parser = subparsers.add_parser(
        "search", help="Searches movie database for query"
    )
    _ = search_parser.add_argument("query", type=str, help="Query to be embedded")
    _ = search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of search results to print",
    )

    return parser


def cmd_verify():
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def cmd_embed_text(text: str):
    search = SemanticSearch()
    return search.generate_embedding(text)


def cmd_verify_embeddings():
    search = SemanticSearch()
    documents = load_movies()

    embeddings = search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def cmd_search(query: str, limit: int) -> list[dict[str, str | float]]:
    search = SemanticSearch()

    documents = load_movies()
    _ = search.load_or_create_embeddings(documents)

    return search.search(query, limit)


def main():
    parser = get_parser()
    args = parser.parse_args()

    match args.command:
        case "verify":
            cmd_verify()

        case "embed_text":
            text = cast(str, args.text)
            embedding = cmd_embed_text(text)

            print(f"Text: {text}")
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Dimensions: {embedding.shape[0]}")

        case "verify_embeddings":
            cmd_verify_embeddings()

        case "embedquery":
            query = cast(str, args.query)
            embedding = cmd_embed_text(query)

            print(f"Query: {query}")
            print(f"First 5 dimensions: {embedding[:5]}")
            print(f"Shape: {embedding.shape}")

        case "search":
            query = cast(str, args.query)
            limit = cast(int, args.limit)
            results = cmd_search(query, limit)

            for i, res in enumerate(results, 1):
                print(
                    f"{i}. {res["title"]} (score: {res["score"]})\n{res["description"]}"
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
