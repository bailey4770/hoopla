# hoopla

A boot.dev guided course exploring Retrieval Augmented Generation (RAG) searching.

The project implements various search algorithms into a database
of movie descriptions. Intended to be used as a search function for
a movie streaming service, for example.
There are also precision, recall, and F1 score metrics available.

## Keyword Search

- Good when user can write a very specific query,
eg. `Paddington` will return Paddington movies
- Bad when user cannot, eg. `funny movie` will return only results where
the words `funny` and `movie` appear in the title or description

- Start by building an inverted index of the database.
- Tokenize query and docs to compare tokens and return results
with higher token matches.
  - Remove stopwords, and 'stem' remaining words to increase relevant matches.
- bm25search command, which compares the
Term Frequency - Inverse Document Frequency (TF-IDF) of the query for each
doc in the database.
  - TF measures how many times a term appears in a doc.
  - IDF measures when a word is very rare in the dataset.
  - A high TF-IDF score represents a term being very common in a doc,
  but rare in the entire dataset.
  - Okapi BM25 smartly combines TF-IDF scores to return the best matches.

### Usage

- Run `uv run Keyword_search_cli.py search <query>` for standard keyword match search
- Run `uv run Keyword_search_cli.py bm25search <query>`
  - Optional `--limit` flag returns custom number of search results.

## Semantic Search

- Good when user is searching for a concept,
eg. `funny movie` will reliably return comedy movies.
- Bad when user is searching for something specific, eg.
`Monsters vs. Aliens` might return movies about aliens and monsters
rather than this specific movie the user is likely searching for.

- Build semantic embeddings database.
  - Pass doc database to a text embedder to encode semantic meanings.
  - This is a computationally heavy process, so we perform this once,
  and store the results.
- Semantically embed user query, and compare cosine similarity between
query embedding and each doc's embedding.
- Splitting our docs into chunks can prevent semantic dilution.
  - Semantic dilution occurs when a doc covers too many concepts to
  be carry all those concepts into just one embedding.
  - By including some overlap in our chunks, we can retain some of the
  context from the original doc.

### How to go further

Currently I store embeddings as a numpy file in a cache dir, and load these
into memory when performing search. This was an easy short-cut in implementation.
However, to make this production ready, this needs to be migrated to a
vector database.

Concepts studied:

- Locality Sensitive Hashing (LSH) - splitting vectors into 'buckets' to increase
search speed, at the potential cost of missing matches
- Hierarchical Navigable Small World (HNSW) - database consists of layers of
increasingly dense graphs. We find the local minimum at each layer to greatly
speed up search.
- Inverted File with Flat Vector (IVF) - vector space is divided in clusters
(similar to genres) and each cluster has a representative node. Comparison to
representative node can greatly reduce search space.

I would love to pursue these concepts further.

### Usage

- Run `uv run semantic_search_cli.py search_chunked <query>`.
  - Optional `--limit` flag returns custom number of search results.

## Hybrid Search

- Keyword search and semantic search both have their pros and cons...
- We can combine their results by normalizing their scores.
- Min-max normalization:
  - Order the results from each search method and normalizing the scores.
  - Combine normalized score from both methods into hybrid score for each doc.
- Reciprocal rank fusion:
  - Better handles major outliers than standard min-max.

### Usage

- Run `uv run hybrid_search_cli.py weighted-search <query>`.
  - Optional `--limit` flag returns custom number of search results.
  - `--alpha` flag controls weight given to keyword search scores.
- Run `uv run hybrid_search_cli.py rrf-search <query>`.
  - Optional `--limit` flag returns custom number of search results.
  - Optional `--k` flag controls weight given to lower/higher scores.
  (low k means low scores given less weight)

## LLM

We can also introduce LLMs to improve our search:

- Enhancing queries by fixing spelling mistakes or improving specificity.
- Reranking results.

### Usage

For all LLM commands, user must store `GEMINI_API_KEY=` in a `.env` file
in the project root.

- Run `uv run hybrid_search_cli.py rrf-search <query>`.
  - Optional `--limit` flag returns custom number of search results.
  - Optional `--k` flag controls weight given to lower/higher scores.
  (low k means low scores given less weight)
  - Optional `--enhance` flag can either spell, rewrite, or expand a user query
  - Optional `--rerank-method` flag can ask an LLM to
  either individually or batch re-rank results

## Cross-Encoder

LLM API calls can be very slow and expensive. A cross-encoder is much cheaper,
faster, and more specialised than a general purpose LLM. It semantically embeds
queries and docs and outputs a single similarity score.

### Usage

- Run `uv run hybrid_search_cli.py rrf-search <query>`.
  - Optional `--limit` flag returns custom number of search results.
  - Optional `--k` flag controls weight given to lower/higher scores.
  (low k means low scores given less weight)
  - Optional `--rerank-method` flag with cross_encoder.

## Metrics

- The first and best way to test the performance of search engine is the 'vibe check'
  - Does the search feel good?
- But when we want numerical metrics to measure performance, precision, recall
and F1 are implemented.
- These require a 'golden dataset' curated by field experts.
- Precision asks 'how many results are actually relevant?'
- Recall asks 'how much of what was relevant was returned?'
- These are often opposed. F1 score combines these scores using harmonic mean.
- There is also the option of asking an LLM to evaluate our search results.

### Usage

- Run `uv run evaluation_cli.py`
  - Optional `--limit` flag determines number of results to evaluate
- Run `uv run hybrid_search_cli.py rrf-search <query>`.
  - Optional `--limit` flag returns custom number of search results.
  - Optional `--k` flag controls weight given to lower/higher scores.
  (low k means low scores given less weight)
  - Optional `--evaluate` flag to ask LLM to evaluate results.

## Retrieval Augmented Generation (RAG) (finally!)

1. **Retrieve** relevant documents using standard search algorithms.
2. **Augment** the LLMs context with the most relevant documents.
3. **Generate** a natural language response to the user query.

Implemented here are the options for standard RAG, summarizing (with citations),
or answering a question.

### Usage

- Run `uv run augmented_generation_cli.py <cmd> <query>` where cmd is one of:
  - `rag`
  - `summarize`
  - `citations`
  - `question`

## Multimodal Search

Also included is the ability to provide an image path and perform a search,
either purely off that image, or with an attached text query. The image with the
attached query is passed to an LLM, who rewrites the query based on the image.
A pure image query, on the other hand, is semantically embedded by a multimodal
model. This model is capable of semantically embedding both images and text.
This is made possible by a contrastive learning approach to its training.

### How to go further

I want to pursue neural network training in a future project, to better grasp how
these models perform semantic embedding.

### Usage

- Run `uv run describe_image_cli.py image_search <image_path>`
  - `--image` should be the path the image
  - `--query` is the query to be answered
- Run `uv run multimodal_search_cli.py image_search <image_path>`
