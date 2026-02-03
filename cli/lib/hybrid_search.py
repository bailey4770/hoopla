from collections import defaultdict

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import Movies, Movie, DEFAULT_SEARCH_LIMIT


class HybridSearch:
    def __init__(self, documents: Movies) -> None:
        self.docmap: dict[int, Movie] = {}
        for doc in documents:
            self.docmap[doc["id"]] = doc

        self.semantic_search: ChunkedSemanticSearch = ChunkedSemanticSearch()
        _ = self.semantic_search.load_or_create_embeddings(documents)

        self.idx: InvertedIndex = InvertedIndex()
        try:
            self.idx.load()
        except FileNotFoundError:
            self.idx.build(documents)
            self.idx.save()
        except Exception as e:
            print("Error loading keyword index: ", e)
            raise e

    def _bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[tuple[int, str, float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results: list[tuple[int, str, float]] = self._bm25_search(
            query, limit * 500
        )
        normalized_bm25_scores: list[float] = normalize_scores(
            [score for _, _, score in bm25_results]
        )
        normalized_bm25_results: dict[int, float] = {
            id: score for (id, _, _), score in zip(bm25_results, normalized_bm25_scores)
        }

        semantic_results: list[dict[str, float]] = self.semantic_search.search_chunks(
            query, limit * 500
        )
        normalized_semantic_scores: list[float] = normalize_scores(
            [res["score"] for res in semantic_results]
        )
        normalized_semantic_results: dict[int, float] = {
            res["id"]: score
            for res, score in zip(semantic_results, normalized_semantic_scores)
        }

        doc_id_to_scores: dict[int, dict[str, str | float]] = defaultdict(dict)

        for id in normalized_bm25_results:
            doc_id_to_scores[id]["bm25"] = normalized_bm25_results[id]

        for id in normalized_semantic_results:
            doc_id_to_scores[id]["semantic"] = normalized_semantic_results[id]

        for id, res in doc_id_to_scores.items():
            doc_id_to_scores[id]["hybrid"] = _hybrid_score(
                res.get("bm25", 0.0), res.get("semantic", 0.0), alpha
            )
            doc_id_to_scores[id]["title"] = self.docmap[id]["title"]
            doc_id_to_scores[id]["description"] = self.docmap[id]["description"]

        return sorted(
            doc_id_to_scores.items(),
            key=lambda _, v: v["hybrid"],
            reverse=True,
        )[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0 for _ in scores]

    return [(s - min_score) / (max_score - min_score) for s in scores]


def _hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score
