"""
import inside func and final decorator instead of type annotation
is done to stop my LSP blowing up and consuming all my memory.
"""

import numpy as np
import re
from sentence_transformers import SentenceTransformer

from .search_utils import Movies, Movie, CACHE_DIR, DEFAULT_MAX_CHUNK_SIZE

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticSearch:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: Movies):
        movie_strings: list[str] = [
            f"{doc['title']}: {doc['description']}" for doc in documents
        ]

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(CACHE_DIR.joinpath("movie_embeddings.npy"), self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: Movies):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        embeddings_path = CACHE_DIR.joinpath("movie_embeddings.npy")
        if not embeddings_path.exists():
            return self.build_embeddings(documents)

        self.embeddings = np.load(embeddings_path)
        if len(self.embeddings) != len(documents):
            raise ValueError

        return self.embeddings

    def search(self, query, limit) -> list[dict[str, float | str]]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        embeddings_query = self.model.encode(query)
        similarity_scores: list[float] = []

        for embedding in self.embeddings:
            similarity_scores.append(cosine_similarity(embeddings_query, embedding))

        if self.documents is None:
            raise ValueError("No documents loaded")
        score_to_doc = zip(similarity_scores, self.documents)

        sorted_scores = sorted(score_to_doc, key=lambda s: s[0], reverse=True)
        top_results = sorted_scores[:limit]
        return [
            {"score": score, "title": doc["title"], "description": doc["description"]}
            for score, doc in top_results
        ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name=DEFAULT_MODEL_NAME) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunked_encodings(self, documents: Movies):
        all_chunks: list[str] = []
        all_chunk_metadata: list[dict[any, any]] = []

        for i, doc in enumerate(documents):
            if doc["description"] == "":
                continue

            chunks = semantic_chunking(doc["description"], 4, 1)
            all_chunks.extend(chunks)

            for chunk in chunks:

                all_chunk_metadata.append()

        return

    def load_or_create_chunk_embeddings(self, documents: Movies) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}


def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_chunking(
    text: str, chunk_size: int = DEFAULT_MAX_CHUNK_SIZE, overlap: int = 0
) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    i = 0
    while i <= len(sentences):
        start = max(0, i - overlap)
        chunk = " ".join(sentences[start : start + chunk_size])
        chunks.append(chunk)
        i = start + chunk_size

    return chunks
