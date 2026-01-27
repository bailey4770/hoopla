from collections import defaultdict
import pickle
from pathlib import Path
import os
import math
from multiprocessing import Pool

from typing import Callable

from .search_utils import (
    process_string,
    Movie,
    Movies,
)

CACHE_DIR = Path("./cache")


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Movie] = {}

    def get_document(self, term: str) -> list[int]:
        doc_ids: set[int] = self.index[term.lower()]
        return sorted(list(doc_ids))

    def build(self, movies: Movies) -> None:
        # build is CPU intensive. We can speed up process by using all cores
        num_workers = os.cpu_count() or 4
        chunk_size = math.ceil(len(movies) / num_workers)
        chunks = [movies[i : i + chunk_size] for i in range(0, len(movies), chunk_size)]

        with Pool(num_workers) as pool:
            partial_indexes: list[dict[str, set[int]]] = pool.map(
                _build_partial_index, chunks
            )

        for pidx in partial_indexes:
            for token, ids in pidx.items():
                self.index[token] |= ids

    def save(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)

        with open(CACHE_DIR.joinpath("index.pkl"), "wb") as f:
            pickle.dump(dict(self.index), f)

        with open(CACHE_DIR.joinpath("docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)


def _build_partial_index(movies_chunk: Movies) -> dict[str, set[int]]:
    processor: Callable[[str], set[str]] = process_string()
    partial_index: dict[str, set[int]] = defaultdict(set)

    for movie in movies_chunk:
        tokens = processor(movie["title"] + " " + movie["description"])
        for token in tokens:
            partial_index[token].add(movie["id"])

    return dict(partial_index)
