import time
import json
import os
from dotenv import load_dotenv
from google import genai
import logging

from .hybrid_search import RRFSearchResult

logger = logging.getLogger(__name__)


def get_gemini_client() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    assert api_key is not None, "could not load gemini api key"

    return genai.Client(api_key=api_key)


def query_gemini(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text if response.text else "no response"


def get_spelling_query(query: str) -> str:
    return f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""


def get_rewritten_query(query: str) -> str:
    return f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""


def get_expanded_query(query: str) -> str:
    return f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""


def rerank_results_individual(
    client: genai.Client,
    query: str,
    docs: list[tuple[int, RRFSearchResult]],
    limit: int,
) -> list[tuple[int, RRFSearchResult]]:
    logger.info("%d results to rerank", len(docs))

    for _, doc in docs:
        prompt = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("document", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Give me ONLY the number in your response, no other text or explanation.

    Score:"""

        new_score = int(query_gemini(client, prompt))
        doc["rerank"] = new_score
        # sleep to avoid gemini rate limit
        # my rate limit is 5 per minute. Wait for 12 seconds.
        time.sleep(12)
        logger.info(
            "%s got new score %d. Waiting 12 seconds before next request",
            doc["title"],
            new_score,
        )

    return sorted(
        docs,
        key=lambda item: item[1]["rrf"],
        reverse=True,
    )[:limit]


def rerank_results_batch(
    client: genai.Client,
    query: str,
    docs: list[tuple[int, RRFSearchResult]],
    limit: int,
) -> list[tuple[int, RRFSearchResult]]:
    logger.info("%d results to rerank. Making api call.", len(docs))

    docs_mapped: dict[int, RRFSearchResult] = {id: doc for id, doc in docs}
    doc_list_str = "\n\n".join(
        f"ID: {id}, Title: {doc['title']}, Description: {doc['description']}"
        for id, doc in docs
    )

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""

    reranked_list_str = query_gemini(client, prompt)
    reranked_list = json.loads(reranked_list_str)
    logger.info("reranked list received and loaded to json")

    reranked_results: list[tuple[int, RRFSearchResult]] = []
    for rank, res_id in enumerate(reranked_list, 1):
        doc = docs_mapped[res_id]
        doc["rerank"] = rank
        reranked_results.append((res_id, doc))

    return reranked_results[:limit]
