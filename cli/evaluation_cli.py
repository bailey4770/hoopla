import argparse
import json

from typing import cast

from hybrid_search_cli import cmd_rrf_search, HybridSearch
from lib.search_utils import load_movies, DEFAULT_K

DEFAULT_GOLDEN_DATASET_PATH = "./data/golden_dataset.json"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    return parser


def get_golden_dataset():
    with open(DEFAULT_GOLDEN_DATASET_PATH, "r") as f:
        golden_dataset = json.load(f)

    return golden_dataset


def main():
    parser = get_parser()

    args = parser.parse_args()
    limit = cast(int, args.limit)

    movie_data = load_movies()
    hybrid_search: HybridSearch = HybridSearch(movie_data)

    golden_dataset = get_golden_dataset()
    for tc in golden_dataset["test_cases"]:
        results = cmd_rrf_search(
            hybrid_search,
            tc["query"],
            DEFAULT_K,
            limit,
        )

        retrieved: list[str] = []
        relevant_retrieved: list[str] = []
        for _, res in results:
            retrieved.append(res["title"])

            if res["title"] in tc["relevant_docs"]:
                relevant_retrieved.append(res["title"])

        precision = len(relevant_retrieved) / len(results)
        recall = len(relevant_retrieved) / len(tc["relevant_docs"])
        f1 = (2 * precision * recall) / (precision + recall)

        print(f"k={limit}\n")
        print(f"- Query: {tc['query']}")
        print(f"    - Precision@{limit}: {precision:.4f}")
        print(f"    - Recall@{limit}: {recall:.4f}")
        print(f"    - F1 Score: {f1:.4f}")
        print(f"    - Retrieved: {retrieved}")
        print(f"    - Relevant: {relevant_retrieved}\n")


if __name__ == "__main__":
    main()
