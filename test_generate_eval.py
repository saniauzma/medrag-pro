# test_generate_eval.py
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from evaluation.testset_generator import TestSetGenerator

generator = TestSetGenerator()

# Generate 10 samples to start — increase later
samples = generator.generate(num_samples=10)

# Preview
print(f"\nGenerated {len(samples)} QA pairs:\n")
for i, s in enumerate(samples, 1):
    print(f"Q{i}: {s.question}")
    print(f"A{i}: {s.ground_truth[:150]}")
    print(f"     Source: {s.source_chunk['source_file']} | Page {s.source_chunk['page_number']}")
    print()

# Save to disk
generator.save(samples)
print("Saved to data/eval_testset.json")