import json
import string

from typing import TypedDict, cast

MOVIES_FILE_PATH = "./data/movies.json"
DEFAULT_SEARCH_LIMIT = 5


class Movie(TypedDict):
    id: int
    title: str
    description: str


Movies = list[Movie]


class MovieData(TypedDict):
    movies: Movies


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def process_string(s: str) -> set[str]:
    to_lower = s.lower()
    punc_removed = to_lower.translate(_PUNCT_TABLE)

    tokens = punc_removed.split()
    for token in tokens:
        if not token:
            del token
    tokens = set(tokens)

    return tokens


def load_movies() -> Movies:
    with open(MOVIES_FILE_PATH, "r") as f:
        movies_data = cast(MovieData, json.load(f))
    return movies_data["movies"]


def print_search_results(query: str, search_results: list[tuple[int, str]]):
    print(f"Searching for: {query}")
    for i, res in enumerate(search_results, 1):
        print(f"{i}. {res[1]}")
