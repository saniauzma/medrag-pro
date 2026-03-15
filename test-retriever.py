# test_retrieval.py
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from ingestion.pipeline import IngestionPipeline
from retrieval.hybrid_retriever import HybridRetriever

# Step 1: Ingest
print("=== INGESTING ===")
pipeline = IngestionPipeline()
result = pipeline.run("data/pdfs/")
print(f"Ingestion: {result}")

# Step 2: Retrieve
print()
print("=== RETRIEVING ===")
retriever = HybridRetriever()
results = retriever.retrieve("large language model healthcare applications")

for i, r in enumerate(results):
    print(f"Rank {i+1} | RRF={r['rrf_score']:.4f} | dense={r['dense_rank']} sparse={r['sparse_rank']}")
    print(f"  page={r['page_number']} section={r['section_title']!r}")
    print(f"  {r['content'][:150]}")
    print()