import json
import re
from typing import override

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, DEFAULT_MAX_CHUNK_SIZE, Movie, Movies

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
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
    while i < len(sentences):
        start = max(0, i - overlap)
        chunk = " ".join(sentences[start : start + chunk_size])
        chunks.append(chunk)
        i = start + chunk_size

    return chunks

class SemanticSearch:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        embedding: np.ndarray = self.model.encode([text])
        return embedding[0]

    def _build_embeddings(self, documents: Movies):
        movie_strings: list[str] = [
            f"{doc['title']}: {doc['description']}" for doc in documents
        ]
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        np.save(CACHE_DIR.joinpath("movie_embeddings.npy"), self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: Movies):
        if not CACHE_DIR.exists():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self.documents = documents
        self.document_map: dict[int, Movie] = {doc["id"]: doc for doc in documents}

        embeddings_path = CACHE_DIR.joinpath("movie_embeddings.npy")
        if not embeddings_path.exists():
            return self._build_embeddings(documents)

        self.embeddings = np.load(embeddings_path)
        if len(self.embeddings) != len(documents):
            raise ValueError("Embeddings count does not match documents")

        return self.embeddings

    def search(self, query: str, limit: int) -> list[dict[str, float | str]]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        embeddings_query = self.model.encode([query], convert_to_numpy=True)[0]
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
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def _build_chunked_encodings(self, documents: Movies):
        all_chunks: list[str] = []
        all_chunk_metadata: list[dict[str, int]] = []

        for doc_idx, doc in enumerate(documents):
            if doc["description"] == "":
                continue

            chunks = semantic_chunking(doc["description"], 4, 1)
            all_chunks.extend(chunks)

            for chunk_idx in range(len(chunks)):
                chunk_metadata = {
                    "movie_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks),
                }
                all_chunk_metadata.append(chunk_metadata)

        print(len(all_chunks))
        assert(len(all_chunks) == 72909)

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = all_chunk_metadata

        chunk_embeddings_path = CACHE_DIR.joinpath("chunk_embeddings.npy")
        np.save(chunk_embeddings_path, self.chunk_embeddings)

        chunk_metadata_path = CACHE_DIR.joinpath("chunk_metadata.json")
        with chunk_metadata_path.open("w") as f:
            json.dump({"chunks": all_chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    @override
    def load_or_create_embeddings(self, documents: Movies):
        CACHE_DIR.mkdir(exist_ok=True)

        self.documents: Movies = documents
        self.document_map: dict[int, Movie] = {doc["id"]: doc for doc in documents}

        chunk_embeddings_path = CACHE_DIR.joinpath("chunk_embeddings.npy")
        if not chunk_embeddings_path.exists():
            return self._build_chunked_encodings(documents) 
            
        chunk_metadata_path = CACHE_DIR.joinpath("chunk_metadata.json")
        if not chunk_metadata_path.exists():
            return self._build_chunked_encodings(documents)

        self.chunk_embeddings = np.load(chunk_embeddings_path)
        self.chunk_metadata = json.load(chunk_metadata_path.open())

        return self.chunk_embeddings