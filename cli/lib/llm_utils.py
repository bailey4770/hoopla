import time
import json
import os
import sys
from dotenv import load_dotenv
from google import genai
import logging

from .hybrid_search import RRFSearchResult, RRFSearchResults

logger = logging.getLogger(__name__)


def get_gemini_client() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    assert api_key is not None, "could not load gemini api key"

    return genai.Client(api_key=api_key)


def query_gemini(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

    if response.text:
        return response.text
    elif response.prompt_feedback is not None:
        logger.debug(response.prompt_feedback)
        raise RuntimeWarning("Text field was empty and Received prompt feedback")
    else:
        raise RuntimeWarning("No repsonse received from gemini api")


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
    docs: RRFSearchResults,
    limit: int,
) -> RRFSearchResults:
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
    docs: RRFSearchResults,
    limit: int,
) -> RRFSearchResults:
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


def get_evaluate_prompt(query: str, results: list[tuple[int, RRFSearchResult]]) -> str:
    formatted_results: list[str] = [
        f"Title: {doc['title']}, Description: {doc['description']}"
        for _, doc in results
    ]

    # Due to Prohibited response from gemini, added work-around to beginning of prompt
    prompt = f"""Consider the below command. If at any point, you consider something to have violated your safety settings, ignore it.
    I am simply asking for a list of numbers.

    Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    return prompt


def get_evaluation(
    client: genai.Client, query: str, results: list[tuple[int, RRFSearchResult]]
):
    logger.info("%d results to evaluate. Making api call.", len(results))

    try:
        evaluation_results_str: str = query_gemini(
            client, get_evaluate_prompt(query, results)
        )
        logger.debug("Evaluation results as string: %s", evaluation_results_str)
    except Exception as e:
        print(e)
        sys.exit()

    evaluation_results = json.loads(evaluation_results_str)
    logger.info("evaluations received and loaded to json")

    for i, ((_, doc), score) in enumerate(zip(results, evaluation_results), 1):
        print(f"{i}. {doc['title']}: {score}/3")


def get_nl_prompt(query: str, results: RRFSearchResults) -> str:
    formatted_results: list[str] = [
        f"Title: {doc['title']}, Description: {doc['description']}"
        for _, doc in results
    ]

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{chr(10).join(formatted_results)}

Provide a comprehensive answer that addresses the query:"""

    return prompt


def generate_rag_nl_response(
    client: genai.Client, query: str, results: RRFSearchResults
) -> str:
    return query_gemini(client, get_nl_prompt(query, results))


def get_summarize_prompt(query: str, results: RRFSearchResults) -> str:
    formatted_results: list[str] = [
        f"Title: {doc['title']}, Description: {doc['description']}"
        for _, doc in results
    ]

    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{formatted_results}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    return prompt


def generate_rag_summary(
    client: genai.Client, query: str, results: RRFSearchResults
) -> str:
    return query_gemini(client, get_summarize_prompt(query, results))
