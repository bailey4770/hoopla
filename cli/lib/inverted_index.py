from collections import defaultdict, Counter
import pickle
from pathlib import Path
import os
import math
from multiprocessing import Pool

from typing import Callable

from .search_utils import (
    process_string,
    Movies,
)

CACHE_DIR = Path("./cache")


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, str] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}

    def get_document(self, term: str) -> list[tuple[int, str]]:
        doc_ids: set[int] = self.index[term.lower()]

        docs: list[str] = []
        for id in doc_ids:
            docs.append(self.docmap[id])

        return sorted(list(zip(doc_ids, docs)))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = process_string()(term)
        if len(tokens) != 1:
            raise ValueError("invalid search term")
        token: str = tokens.pop()

        return self.term_frequencies[doc_id][token]

    def get_bm25_idf(self, term: str) -> float:
        tokens = process_string()(term)
        if len(tokens) != 1:
            raise ValueError("invalid search term")
        token: str = tokens.pop()

        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(token, set()))

        numerator = total_doc_count - term_match_doc_count + 0.5
        denominator = term_match_doc_count + 0.5

        return math.log(numerator / denominator + 1)

    def build(self, movies: Movies) -> None:
        # build is CPU intensive. We can speed up process by using all cores
        num_workers = os.cpu_count() or 4
        chunk_size = math.ceil(len(movies) / num_workers)
        chunks = [movies[i : i + chunk_size] for i in range(0, len(movies), chunk_size)]

        with Pool(num_workers) as pool:
            partial_builds: list[
                tuple[dict[str, set[int]], dict[int, Counter[str]]]
            ] = pool.map(_build_partial_index, chunks)

        partial_indexes: list[dict[str, set[int]]] = []
        for pb in partial_builds:
            partial_indexes.append(pb[0])
            self.term_frequencies |= pb[1]

        for pidx in partial_indexes:
            for token, ids in pidx.items():
                self.index[token] |= ids

        for movie in movies:
            self.docmap[movie["id"]] = movie["title"]

    def save(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)

        with open(CACHE_DIR.joinpath("index.pkl"), "wb") as f:
            pickle.dump(dict(self.index), f)

        with open(CACHE_DIR.joinpath("docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)

        with open(CACHE_DIR.joinpath("term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        index_path = CACHE_DIR.joinpath("index.pkl")
        if not index_path.exists():
            raise FileNotFoundError()

        docmap_path = CACHE_DIR.joinpath("docmap.pkl")
        if not docmap_path.exists():
            raise FileNotFoundError()

        term_frequencies_path = CACHE_DIR.joinpath("term_frequencies.pkl")
        if not term_frequencies_path.exists():
            raise FileNotFoundError()

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)


def _build_partial_index(
    movies_chunk: Movies,
) -> tuple[dict[str, set[int]], dict[int, Counter[str]]]:
    processor: Callable[[str], list[str]] = process_string()

    partial_index: dict[str, set[int]] = defaultdict(set)
    partial_tf: dict[int, Counter[str]] = {}

    for movie in movies_chunk:
        id = movie["id"]
        tokens = processor(movie["title"] + " " + movie["description"])
        for token in tokens:
            partial_index[token].add(id)

        partial_tf[id] = Counter(tokens)

    return dict(partial_index), partial_tf
