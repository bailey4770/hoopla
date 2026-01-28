#!/usr/bin/env python3

import argparse
import json
from typing import cast

from lib.semantic_search import SemanticSearch
from lib.search_utils import load_movies


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

    return parser


def cmd_verify():
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def cmd_embed_text(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def cmd_verify_embeddings():
    search = SemanticSearch()
    documents = load_movies()

    embeddings = search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def main():
    parser = get_parser()
    args = parser.parse_args()

    match args.command:
        case "verify":
            cmd_verify()

        case "embed_text":
            text = cast(str, args.text)
            cmd_embed_text(text)

        case "verify_embeddings":
            cmd_verify_embeddings()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
