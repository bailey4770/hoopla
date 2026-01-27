import json
import string
from nltk.stem import PorterStemmer

from typing import TypedDict, cast

MOVIES_FILE_PATH = "./data/movies.json"
STOPWORDS_FILE_PATH = "./data/stopwords.txt"
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
DEFAULT_SEARCH_LIMIT = 5


class Movie(TypedDict):
    id: int
    title: str
    description: str


Movies = list[Movie]


class MovieData(TypedDict):
    movies: Movies


def process_string(s: str, stopwords: set[str]) -> set[str]:
    to_lower = s.lower()
    punc_removed = to_lower.translate(_PUNCT_TABLE)

    tokens = punc_removed.split()

    removed_stopwords: set[str] = set()
    for token in tokens:
        if token and token not in stopwords:
            removed_stopwords.add(token)

    stemmer = PorterStemmer()
    stemmed_tokens: set[str] = set(map(stemmer.stem, removed_stopwords))

    return stemmed_tokens


def load_movies() -> Movies:
    with open(MOVIES_FILE_PATH, "r") as f:
        movies_data = cast(MovieData, json.load(f))
    return movies_data["movies"]


def print_search_results(query: str, search_results: list[tuple[int, str]]):
    print(f"Searching for: {query}")
    for i, res in enumerate(search_results, 1):
        print(f"{i}. {res[1]}")


def load_stopwords() -> set[str]:
    with open(STOPWORDS_FILE_PATH, "r") as f:
        stopwords = f.read().splitlines()
    return set(stopwords)
