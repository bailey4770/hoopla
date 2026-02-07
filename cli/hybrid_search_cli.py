from typing import cast, Callable
import argparse
import os
from dotenv import load_dotenv
from google import genai

from lib.hybrid_search import HybridSearch, normalize_scores
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_K,
    DOC_PREVIEW_LENGTH,
    load_movies,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize")
    _ = normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="+",
        help="Scores to normalize using min-max normalization",
    )

    weighted_search_parser = subparsers.add_parser("weighted-search")
    _ = weighted_search_parser.add_argument("query", type=str, help="Search query")
    _ = weighted_search_parser.add_argument(
        "--alpha", type=float, help="Weight for BM25 scores (0.0 to 1.0)"
    )
    _ = weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of top results to return",
    )

    rrf_search_parser = subparsers.add_parser("rrf-search")
    _ = rrf_search_parser.add_argument("query", type=str, help="Search query")
    _ = rrf_search_parser.add_argument(
        "--k",
        type=float,
        default=DEFAULT_K,
        help="Weight given to higher vs lower ranked scores",
    )
    _ = rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of top results to return",
    )
    _ = rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite"],
        help="Query enhancement method",
    )

    return parser


def cmd_normalize(scores: list[float]) -> list[float]:
    return normalize_scores(scores)


def cmd_weighted_search(
    hybrid_search: HybridSearch, query: str, alpha: float, limit: int
):
    return hybrid_search.weighted_search(query, alpha, limit)


def cmd_rrf_search(
    hybrid_search: HybridSearch,
    query: str,
    k: float,
    limit: int,
    enhance_method: str = "",
):
    if enhance_method:
        match enhance_method:
            case "spell":
                enhanced_query = enhance_query(get_spelling_prompt(query))
                print(
                    f"Enhanced query ({enhance_method}): '{query}' -> '{enhanced_query}'"
                )
            case "rewrite":
                enhanced_query = enhance_query(get_rewritten_prompt(query))
                print(
                    f"Enhanced query ({enhance_method}): '{query}' -> '{enhanced_query}'"
                )
            case _:
                raise ValueError("unrecognised enhance method")

    else:
        enhanced_query = query

    return hybrid_search.rrf_search(enhanced_query, k, limit)


def enhance_query(prompt: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text if response.text else "no response"


def get_spelling_prompt(query: str) -> str:
    return f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""


def get_rewritten_prompt(query: str) -> str:
    return f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    movie_data = load_movies()
    hybrid_search: HybridSearch = HybridSearch(movie_data)

    query: str = cast(str, args.query)
    limit: int = cast(int, args.limit)

    match args.command:
        case "normalize":
            scores: list[float] = cast(list[float], args.scores)
            normalized_scores: list[float] = cmd_normalize(scores)

            for s in normalized_scores:
                print(f"* {s:.4f}")

        case "weighted-search":
            alpha: float = cast(float, args.alpha)

            results = cmd_weighted_search(hybrid_search, query, alpha, limit)
            for i, (_, res) in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                print(f"     Hybrid Score: {res['hybrid']:.4f}")
                print(f"     BM25: {res['bm25']:.4f}, Semantic: {res['semantic']:.4f}")
                print(
                    f"     Description: {res['description'][:DOC_PREVIEW_LENGTH].replace('\n', ' ')}..."
                )

        case "rrf-search":
            k: float = cast(float, args.k)
            enhance_method: str = cast(str, args.enhance)

            results = cmd_rrf_search(hybrid_search, query, k, limit, enhance_method)
            for i, (_, res) in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                print(f"     RRF Score: {res['rrf']:.4f}")
                print(
                    f"     BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}"
                )
                print(
                    f"     Description: {res['description'][:DOC_PREVIEW_LENGTH].replace('\n', ' ')}..."
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
