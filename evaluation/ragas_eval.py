# evaluation/ragas_eval.py
# ------------------------
# Evaluates MedRAG Pro using RAGAS metrics.
#
# Four metrics measured:
#
#   Faithfulness     — does the answer contain only info from retrieved context?
#                      Score 1.0 = every claim is grounded in context
#                      Score 0.0 = answer ignores context completely
#
#   Answer Relevancy — does the answer actually address the question asked?
#                      Score 1.0 = answer is directly relevant
#                      Score 0.0 = answer is off-topic
#
#   Context Precision — of retrieved chunks, how many were actually useful?
#                       Score 1.0 = every retrieved chunk contributed
#                       Score 0.0 = retrieved chunks were irrelevant
#
#   Context Recall   — did we retrieve all information needed to answer?
#                      Score 1.0 = retrieved everything necessary
#                      Score 0.0 = missed critical information
#
# RAGAS runs these metrics using an LLM-as-judge pattern —
# it uses Ollama to score each metric, so no external API needed.

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset
# HuggingFace Dataset — RAGAS requires this specific format.
# We convert our EvalSample list into a Dataset object.

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
# These four are the standard RAGAS metrics.
# Each is a callable that scores one aspect of RAG quality.

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# RAGAS needs an LLM and an embedding model to run its scoring.
# These wrappers let us plug in our local Ollama LLM and bge-m3.

from langchain_ollama import ChatOllama, OllamaEmbeddings
# OllamaEmbeddings — we use this for RAGAS embedding scoring
# (separate from our bge-m3 embedder to avoid VRAM conflicts)

from config import settings
from evaluation.testset_generator import EvalSample, TestSetGenerator
from generation.rag_chain import MedRAGChain
from ragas.run_config import RunConfig


logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    Evaluates the full RAG pipeline using RAGAS metrics.

    Workflow:
      1. Load test set (or generate if not exists)
      2. Run each question through MedRAGChain
      3. Collect answers + retrieved contexts
      4. Score with RAGAS (LLM-as-judge)
      5. Report metrics + save results

    Usage:
        evaluator = RAGASEvaluator()
        results = evaluator.run(num_samples=10)
        evaluator.print_report(results)
    """

    def __init__(self):
        # ── RAG chain — what we're evaluating ─────────────────
        logger.info("Initializing RAG chain for evaluation...")
        self.chain = MedRAGChain()

        # ── RAGAS LLM judge — uses Ollama locally ─────────────
        # RAGAS uses an LLM to judge faithfulness, relevancy etc.
        # We wrap our local Ollama so RAGAS doesn't need OpenAI.
        ragas_llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            # Deterministic scoring — same question always gets same score
        )
        self.ragas_llm = LangchainLLMWrapper(ragas_llm)
        # LangchainLLMWrapper adapts LangChain LLM to RAGAS interface

        # ── RAGAS embeddings — for answer relevancy metric ─────
        # Answer relevancy uses embeddings to measure semantic similarity.
        # We use OllamaEmbeddings here (lighter than bge-m3 for eval).
        ragas_embeddings = OllamaEmbeddings(
            model="llama3.2:3b",
            # Using the same model for embeddings in RAGAS eval.
            # Not as accurate as bge-m3 but avoids VRAM conflicts
            # since bge-m3 is already loaded for retrieval.
            base_url=settings.ollama_base_url,
        )
        self.ragas_embeddings = LangchainEmbeddingsWrapper(ragas_embeddings)

        # ── Test set generator — for loading/generating samples ─
        self.generator = TestSetGenerator()

        logger.info("RAGASEvaluator ready ✅")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        num_samples: int = 10,
        testset_path: str = "data/eval_testset.json",
    ) -> dict:
        """
        Run the full evaluation pipeline.

        Args:
            num_samples:   How many QA pairs to evaluate
                           More = more reliable metrics, slower
            testset_path:  Path to saved test set JSON
                           If not found, generates a new one

        Returns:
            Dict with RAGAS scores + per-sample results:
            {
                "faithfulness": 0.87,
                "answer_relevancy": 0.82,
                "context_precision": 0.79,
                "context_recall": 0.71,
                "num_samples": 10,
                "samples": [...]
            }
        """
        # ── Step 1: Load or generate test set ─────────────────
        samples = self._load_or_generate(testset_path, num_samples)
        samples = samples[:num_samples]
        # Slice to requested size — test set may have more samples

        logger.info(f"Evaluating {len(samples)} samples...")

        # ── Step 2: Run RAG chain on each question ─────────────
        samples = self._run_rag_on_samples(samples)

        # ── Step 3: Build RAGAS dataset ────────────────────────
        dataset = self._build_ragas_dataset(samples)

        # ── Step 4: Score with RAGAS ───────────────────────────
        logger.info("Running RAGAS scoring...")
        scores = self._score(dataset)

        # ── Step 5: Save + return results ─────────────────────
        results = {
            **scores,
            "num_samples": len(samples),
            "samples": [s.to_dict() for s in samples],
        }

        self._save_results(results)
        return results

    def print_report(self, results: dict) -> None:
        """Print a formatted evaluation report."""
        print("\n" + "=" * 60)
        print("RAGAS EVALUATION REPORT")
        print("=" * 60)
        print(f"Samples evaluated: {results['num_samples']}")
        print()

        metrics = [
            ("Faithfulness",      "faithfulness",      0.85),
            ("Answer Relevancy",  "answer_relevancy",  0.80),
            ("Context Precision", "context_precision", 0.75),
            ("Context Recall",    "context_recall",    0.70),
        ]

        print(f"{'Metric':<22} {'Score':>6}  {'Target':>8}  {'Status'}")
        print("-" * 55)

        for name, key, target in metrics:
            score = results.get(key, 0.0)
            status = "✅ PASS" if score >= target else "❌ FAIL"
            print(f"{name:<22} {score:>6.3f}  {target:>8.2f}  {status}")

        print("=" * 60)

        # Overall pass/fail
        all_pass = all(
            results.get(key, 0) >= target
            for _, key, target in metrics
        )
        print(f"Overall: {'✅ PRODUCTION READY' if all_pass else '⚠️  NEEDS IMPROVEMENT'}")
        print("=" * 60)

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_or_generate(
        self,
        path: str,
        num_samples: int,
    ) -> list[EvalSample]:
        """Load test set from disk, or generate if not found."""
        try:
            samples = self.generator.load(path)
            logger.info(f"Loaded {len(samples)} samples from {path}")
            return samples
        except FileNotFoundError:
            logger.info("Test set not found — generating...")
            samples = self.generator.generate(num_samples)
            self.generator.save(samples, path)
            return samples

    def _run_rag_on_samples(
        self,
        samples: list[EvalSample],
    ) -> list[EvalSample]:
        """
        Run each question through the RAG chain.
        Fills in sample.answer and sample.contexts.
        """
        for i, sample in enumerate(samples):
            logger.info(
                f"Running RAG {i+1}/{len(samples)}: "
                f"'{sample.question[:60]}'"
            )
            try:
                response = self.chain.query(sample.question)

                sample.answer = response.answer
                # The LLM's actual answer

                sample.contexts = [
                    chunk["content"]
                    for chunk in response.sources
                ]
                # The text of each retrieved chunk.
                # RAGAS uses these to measure faithfulness and precision.

            except Exception as e:
                logger.error(f"RAG chain failed for sample {i+1}: {e}")
                sample.answer = "Error generating answer"
                sample.contexts = []

        return samples

    def _build_ragas_dataset(
        self,
        samples: list[EvalSample],
    ) -> Dataset:
        """
        Convert EvalSample list to HuggingFace Dataset.

        RAGAS expects a Dataset with these exact column names:
          question      → the question asked
          answer        → the RAG system's answer
          contexts      → list of retrieved chunk texts
          ground_truth  → the correct answer

        Column names must match exactly — RAGAS checks for them.
        """
        data = {
            "question":     [s.question      for s in samples],
            "answer":       [s.answer        for s in samples],
            "contexts":     [s.contexts      for s in samples],
            "ground_truth": [s.ground_truth  for s in samples],
        }
        return Dataset.from_dict(data)

    def _score(self, dataset: Dataset) -> dict:
        """
        Run RAGAS evaluation on the dataset.
        Returns a dict of metric_name → float score.
        """
        from ragas.run_config import RunConfig

        try:
            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                raise_exceptions=False,
                run_config=RunConfig(
                    max_workers=1,
                    # Force sequential execution — one request at a time.
                    # Ollama can't handle parallel requests — this prevents
                    # all the TimeoutError jobs we saw.

                    timeout=120,
                    # 2 minute timeout per scoring call.
                    # llama3.2:3b on GPU takes ~20s — 120s is safe.

                    max_retries=2,
                    # Retry failed jobs twice before giving up.
                ),
            )

            # ── RAGAS v0.2+ result parsing ────────────────────────
            # result is a RAGAS EvaluationResult object.
            # Convert to pandas DataFrame first, then extract means.
            df = result.to_pandas()

            def safe_mean(col: str) -> float:
                """Extract mean score, handling NaN and missing columns."""
                if col not in df.columns:
                    return 0.0
                values = df[col].dropna()
                # dropna() removes NaN entries from failed scorings
                return float(values.mean()) if len(values) > 0 else 0.0

            return {
                "faithfulness":      safe_mean("faithfulness"),
                "answer_relevancy":  safe_mean("answer_relevancy"),
                "context_precision": safe_mean("context_precision"),
                "context_recall":    safe_mean("context_recall"),
            }

        except Exception as e:
            logger.error(f"RAGAS scoring failed: {e}")
            # Log the full traceback for debugging
            import traceback
            logger.error(traceback.format_exc())
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }
        
    def _save_results(self, results: dict) -> None:
        """Save evaluation results to disk."""
        path = Path("data/eval_results.json")
        # Save without the full sample data to keep file small
        summary = {k: v for k, v in results.items() if k != "samples"}
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved → {path}")