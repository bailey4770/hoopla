import json
from pathlib import Path
import string
from typing import Callable, TypedDict, cast, Any

from nltk.stem import PorterStemmer


class Movie(TypedDict):
    id: int
    title: str
    description: str


Movies = list[Movie]


class MovieData(TypedDict):
    movies: Movies


MOVIES_FILE_PATH = "./data/movies.json"
STOPWORDS_FILE_PATH = "./data/stopwords.txt"
CACHE_DIR = Path("./cache")

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

DEFAULT_SEARCH_LIMIT = 5
DOC_PREVIEW_LENGTH = 100
SCORE_PRECISION = 4

DEFAULT_CHUNK_SIZE = 200
DEFAULT_MAX_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 2

BM25_K1 = 1.5
BM25_B = 0.75


class SimilarityScore(TypedDict):
    chunk_idx: int
    movie_id: int
    score: float


def process_string() -> Callable[[str], list[str]]:
    def _load_stopwords() -> set[str]:
        with open(STOPWORDS_FILE_PATH, "r") as f:
            stopwords = f.read().splitlines()
        return set(stopwords)

    stopwords = _load_stopwords()
    stemmer = PorterStemmer()

    def wrapper(s: str) -> list[str]:
        punc_removed = s.lower().translate(_PUNCT_TABLE)
        tokens = punc_removed.split()

        filtered = [t for t in tokens if t and t not in stopwords]
        return [stemmer.stem(t) for t in filtered]

    return wrapper


def load_movies() -> Movies:
    with open(MOVIES_FILE_PATH, "r") as f:
        movies_data = cast(MovieData, json.load(f))
    return movies_data["movies"]


def print_search_results(query: str, search_results: list[tuple[int, str]]) -> None:
    print(f"Searching for: {query}")
    for i, res in enumerate(search_results, 1):
        print(f"{i}. {res[1]}")


def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
