import argparse
import logging
from enum import Enum

from typing import cast, Callable
from google import genai


from lib.hybrid_search import HybridSearch, RRFSearchResults
from lib.search_utils import load_movies
from hybrid_search_cli import cmd_rrf_search
from lib.llm_utils import (
    generate_rag_nl_response,
    generate_rag_summary,
    get_gemini_client,
)

logger = logging.getLogger(__name__)

RAGFunction = Callable[[genai.Client, str, RRFSearchResults], str]


class RAGMethod(Enum):
    NL_RESPONSE: RAGFunction = generate_rag_nl_response
    SUMMARIZE: RAGFunction = generate_rag_summary


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Get summary (search + generate summary)"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")

    return parser


def cmd_rag(query: str, rag_method: RAGFunction) -> tuple[RRFSearchResults, str]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    gemini_client = get_gemini_client()

    results = cmd_rrf_search(hybrid_search, query, limit=5)
    logger.info("rrf search results received")

    rag_response = rag_method(gemini_client, query, results)
    logger.info("rag response generated")

    return results, rag_response


def main():
    parser = get_parser()
    args = parser.parse_args()

    query = cast(str, args.query)

    match args.command:
        case "rag":
            results, rag_response = cmd_rag(query, RAGMethod.NL_RESPONSE)

            print("Search Results:")
            for _, res in results:
                print(" - ", res["title"])

            print("\n RAG Response:\n", rag_response)

        case "summarize":
            results, summary = cmd_rag(query, RAGMethod.SUMMARIZE)

            print("Search Results:")
            for _, res in results:
                print(" - ", res["title"])

            print("\n LLM Summary:\n", summary)

        case _:
            parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
