"""
import inside func and final decorator instead of type annotation
is done to stop my LSP blowing up and consuming all my memory.
"""

import numpy as np
from typing import final

from .search_utils import Movies, CACHE_DIR


def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@final
class SemanticSearch:
    def __init__(self):
        self.model = get_model()
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: Movies):
        movie_strings: list[str] = []
        for doc in documents:
            movie_strings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(CACHE_DIR.joinpath("movie_embeddings.npy"), self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: Movies):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        embeddings_path = CACHE_DIR.joinpath("movie_embeddings.npy")
        if not embeddings_path.exists():
            return self.build_embeddings(documents)

        self.embeddings = np.load(embeddings_path)
        if len(self.embeddings) != len(documents):
            raise ValueError

        return self.embeddings
