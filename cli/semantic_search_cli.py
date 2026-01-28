#!/usr/bin/env python3

import argparse

from lib.semantic_search import SemanticSearch


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser("verify", help="Verify loaded model")

    return parser


def cmd_verify():
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    match args.command:
        case "verify":
            cmd_verify()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
