# test_retrieval.py
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from ingestion.pipeline import IngestionPipeline
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import MedReranker

QUERY = "what are the challenges of large language models in healthcare"

# Step 1: Ingest
print("=== INGESTING ===")
pipeline = IngestionPipeline()
result = pipeline.run("data/pdfs/")
print(f"Result: {result}")
print()

# Step 2: Retrieve — fetch more candidates for reranker to work with
print("=== RETRIEVING (top 10 for reranker) ===")
retriever = HybridRetriever()
candidates = retriever.retrieve(QUERY, k=10)
print(f"Got {len(candidates)} candidates")
print()

# Step 3: Rerank
print("=== RERANKING ===")
reranker = MedReranker()
results = reranker.rerank(QUERY, candidates, top_k=5)
print()

# Step 4: Show final results
print("=== FINAL TOP 5 ===")
for i, r in enumerate(results):
    print(f"Rank {i+1} | rerank={r['rerank_score']:.3f} | rrf={r['rrf_score']:.4f}")
    print(f"  page={r['page_number']} section={r['section_title']!r}")
    print(f"  {r['content'][:200]}")
    print()