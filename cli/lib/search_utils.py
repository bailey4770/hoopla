import json
import string
from nltk.stem import PorterStemmer

from typing import TypedDict, cast


class Movie(TypedDict):
    id: int
    title: str
    description: str


Movies = list[Movie]


class MovieData(TypedDict):
    movies: Movies


MOVIES_FILE_PATH = "./data/movies.json"
STOPWORDS_FILE_PATH = "./data/stopwords.txt"
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
DEFAULT_SEARCH_LIMIT = 5


def process_string():
    def _load_stopwords() -> set[str]:
        with open(STOPWORDS_FILE_PATH, "r") as f:
            stopwords = f.read().splitlines()
        return set(stopwords)

    stopwords = _load_stopwords()
    stemmer = PorterStemmer()

    def wrapper(s: str) -> set[str]:
        punc_removed = s.lower().translate(_PUNCT_TABLE)
        tokens = punc_removed.split()

        filtered = {t for t in tokens if t and t not in stopwords}
        return {stemmer.stem(t) for t in filtered}

    return wrapper


def load_movies() -> Movies:
    with open(MOVIES_FILE_PATH, "r") as f:
        movies_data = cast(MovieData, json.load(f))
    return movies_data["movies"]


def print_search_results(query: str, search_results: list[tuple[int, str]]):
    print(f"Searching for: {query}")
    for i, res in enumerate(search_results, 1):
        print(f"{i}. {res[1]}")
