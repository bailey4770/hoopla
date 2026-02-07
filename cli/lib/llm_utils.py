import time
import os
from dotenv import load_dotenv
from google import genai

from .hybrid_search import RRFSearchResult


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


def rerank_results(
    client: genai.Client,
    query: str,
    docs: list[tuple[int, RRFSearchResult]],
    limit: int,
) -> list[tuple[int, RRFSearchResult]]:
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

    return sorted(
        docs,
        key=lambda item: item[1]["rrf"],
        reverse=True,
    )[:limit]
