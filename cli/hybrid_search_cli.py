from typing import cast
import argparse
import logging

from lib.hybrid_search import HybridSearch, normalize_scores, RRFSearchResult
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_K,
    DOC_PREVIEW_LENGTH,
    load_movies,
)
from lib.llm_utils import (
    get_gemini_client,
    query_gemini,
    get_spelling_query,
    get_rewritten_query,
    get_expanded_query,
    rerank_results_individual,
    rerank_results_batch,
)
from lib.cross_encoder import rerank_cross_encoder


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
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    _ = rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Query rerank method",
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
    rerank_method: str = "",
):
    client = get_gemini_client()

    if enhance_method:
        match enhance_method:
            case "spell":
                enhanced_query = query_gemini(client, get_spelling_query(query))
                print(
                    f"Enhanced query ({enhance_method}): '{query}' -> '{enhanced_query}'"
                )
            case "rewrite":
                enhanced_query = query_gemini(client, get_rewritten_query(query))
                print(
                    f"Enhanced query ({enhance_method}): '{query}' -> '{enhanced_query}'"
                )
            case "expand":
                enhanced_query = query_gemini(client, get_expanded_query(query))
                print(
                    f"Enhanced query ({enhance_method}): '{query}' -> '{enhanced_query}'"
                )
            case _:
                raise ValueError("unrecognised enhance method")

    else:
        enhanced_query = query

    if rerank_method:
        higher_limit = limit * 5
        fast_results: list[tuple[int, RRFSearchResult]] = hybrid_search.rrf_search(
            enhanced_query, k, higher_limit
        )

        match rerank_method:
            case "individual":
                return rerank_results_individual(
                    client, enhanced_query, fast_results, limit
                )
            case "batch":
                return rerank_results_batch(client, enhanced_query, fast_results, limit)
            case "cross_encoder":
                return rerank_cross_encoder(query, fast_results, limit)
            case _:
                raise ValueError("unrecognised rerank method")

    return hybrid_search.rrf_search(enhanced_query, k, limit)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
            rerank_method: str = cast(str, args.rerank_method)

            results = cmd_rrf_search(
                hybrid_search, query, k, limit, enhance_method, rerank_method
            )
            for i, (_, res) in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                if res["rerank"] != -1:
                    print(f"     Rerank Score: {res['rerank']}")
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
