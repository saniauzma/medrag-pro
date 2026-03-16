# test_rag.py
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from ingestion.pipeline import IngestionPipeline
from generation.rag_chain import MedRAGChain

# Step 1: Ingest (skip if already done)
print("=== INGESTING ===")
pipeline = IngestionPipeline()
result = pipeline.run("data/pdfs/")
print(f"Result: {result}")
print()

# Step 2: Ask a question
print("=== QUERYING ===")
chain = MedRAGChain()

response = chain.query(
    "What are the main challenges of large language models in healthcare?"
)

print()
print("=" * 60)
print("ANSWER:")
print("=" * 60)
print(response.answer)

print()
print("=" * 60)
print("SOURCES USED:")
print("=" * 60)
for i, src in enumerate(response.sources, 1):
    print(f"[{i}] {src['source_file']} | Page {src['page_number']} | {src['section_title']}")
    print(f"     rerank={src['rerank_score']:.3f} | {src['content'][:100]}...")
    print()

print("=" * 60)
print(f"Timing: retrieval={response.retrieval_time:.2f}s | "
      f"generation={response.generation_time:.2f}s | "
      f"total={response.total_time:.2f}s")