# RAG Retrieval Study

This note captures the real-PDF retrieval checks performed on `artificial_intelligence_tutorial.pdf`.

## Questions And Answers

1. Can the retriever consistently find the correct chunk?
   - Yes, for direct definition-style questions about AI. The top result for `What is Artificial Intelligence?` consistently returned the expected chunk across the tested chunk sizes.
   - For weaker or broader queries, retrieval still returns relevant chunks, but not always the exact same one.

2. Does changing chunk size improve or worsen retrieval?
   - In these tests, `500` looked like the best balance.
   - `300` produced many more chunks and slower ingestion.
   - `800` reduced the number of chunks further, but direct retrieval quality was slightly less stable for the AI definition query.

3. What happens if the query uses synonyms?
   - The retriever handled synonyms reasonably well.
   - Example: `What is AI?` still retrieved the same definition chunk and produced strong similarity scores.

4. How long does retrieval take?
   - Retrieval was typically around `10-18 ms` per query on the local machine after indexing.
   - The exact value varies with chunk size and whether the embedding model is already loaded.

5. How many chunks are searched?
   - The API asks Chroma for `top_k=3` results.
   - The search corpus is the full indexed collection, and the retriever returns the 3 nearest chunks.
   - In the experiment, the indexed collection size was `404` chunks at size `300`, `227` chunks at size `500`, and `140` chunks at size `800`.

6. What happens on hallucination tests?
   - Unrelated questions like `Who invented Facebook?`, `What is quantum computing?`, and `How many moons does Mars have?` return `I don't know.` instead of fabricating an answer.
   - The answer pipeline also returns the top matching chunks as sources, so you can inspect what the retriever found even when the LLM declines to answer.

## Measured Results

| Chunk size | Overlap | Chunks | Avg. chunk size | Ingest time | Direct AI query time | Direct AI top score | Synonym query top score |
| ---------- | ------- | -----: | --------------: | ----------: | -------------------: | ------------------: | ----------------------: |
| 300        | 75      |    404 |          299.36 |     13.21 s |             12.11 ms |              0.8221 |                  0.7740 |
| 500        | 100     |    227 |          499.44 |      9.94 s |             11.48 ms |              0.8116 |                  0.7818 |
| 800        | 150     |    140 |          797.50 |      9.91 s |             17.62 ms |              0.7215 |                  0.7477 |

## Takeaway

For this document, `500` is the best default so far. It keeps retrieval strong while reducing the number of stored chunks compared with `300`. The `800` setting is still usable, but it was not clearly better for relevance.

## Multi-Document Support

- The collection can store chunks from multiple PDFs.
- Each chunk stores `filename`, `page`, and `chunk_id` metadata.
- This makes it possible to trace answers back to a specific page in a specific file.
