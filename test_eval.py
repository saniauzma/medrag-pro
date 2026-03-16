# test_eval.py
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from evaluation.ragas_eval import RAGASEvaluator

evaluator = RAGASEvaluator()

# Run on 5 samples first — fast sanity check
# Increase to 20+ for reliable metrics
results = evaluator.run(num_samples=5)

evaluator.print_report(results)