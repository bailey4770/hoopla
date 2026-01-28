"""
import inside func and final decorator instead of type annotation
is done to stop my LSP blowing up and consuming all my memory.
"""

from typing import final


def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@final
class SemanticSearch:
    def __init__(self):
        self.model = get_model()

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        embedding = self.model.encode([text])
        return embedding[0]
