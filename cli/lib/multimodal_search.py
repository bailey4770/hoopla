import logging
import pathlib

from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL_NAME = "clip-ViT-B-32"

logger = logging.getLogger(__name__)


def verify_image_embedding(image_path: pathlib.Path) -> None:
    mm_search = MultimodalSearch()

    embedding = mm_search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


class MultimodalSearch:
    def __init__(self, model_name=MODEL_NAME):
        self.model: SentenceTransformer = SentenceTransformer(model_name)

    def embed_image(self, image_path: pathlib.Path):
        with Image.open(image_path) as im:
            # must pass to method as list, and we only want first (and only) element from response
            encoded_image = self.model.encode([im])[0]  # type: ignore[arg-type]
        logger.info("Image encoded")
        return encoded_image
