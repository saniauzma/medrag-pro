# evaluation/testset_generator.py
# --------------------------------
# Generates synthetic QA pairs from indexed chunks.
#
# Process:
#   1. Sample chunks from the corpus
#   2. Send each to Ollama with a generation prompt
#   3. Parse question + answer from the response
#   4. Save as a dataset for RAGAS evaluation
#
# Output: List[EvalSample] — each has question, ground_truth, source_chunk

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import settings
from retrieval.bm25_index import BM25Index

logger = logging.getLogger(__name__)


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """
    A single evaluation sample for RAGAS.

    RAGAS needs:
      question      → what was asked
      ground_truth  → the correct answer (from the source chunk)
      answer        → what our RAG system actually answered
      contexts      → the chunks our RAG retrieved

    We generate question + ground_truth here.
    answer + contexts get filled in during evaluation.
    """
    question: str
    ground_truth: str
    source_chunk: dict          # the chunk this QA was generated from
    answer: str = ""           # filled during eval run
    contexts: list[str] = field(default_factory=list)  # filled during eval run

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "answer": self.answer,
            "contexts": self.contexts,
            "source_file": self.source_chunk.get("source_file", ""),
            "source_page": self.source_chunk.get("page_number", 0),
        }


# ─── Generator ────────────────────────────────────────────────────────────────

class TestSetGenerator:
    """
    Generates synthetic QA pairs from indexed document chunks.

    Uses Ollama (llama3.2:3b) to generate realistic medical questions
    and their ground truth answers from document chunks.

    Usage:
        generator = TestSetGenerator()
        samples = generator.generate(num_samples=20)
        generator.save(samples, "data/eval_testset.json")
    """

    # Prompt that instructs the LLM to generate a QA pair from a chunk
    GENERATION_PROMPT = """You are creating evaluation data for a medical RAG system.

Given the following text from a medical document, generate ONE specific question
that can be answered using ONLY this text, and the corresponding answer.

RULES:
1. The question must be specific and answerable from the text alone
2. The answer must be a direct, factual response using only the text
3. Avoid yes/no questions — ask for specific information
4. Focus on clinically relevant information (findings, methods, results, numbers)
5. Return ONLY valid JSON in this exact format, nothing else:

{{"question": "your question here", "answer": "your answer here"}}

TEXT:
{chunk_content}

JSON:"""

    def __init__(self):
        self.llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
            # Slightly higher temperature than 0.0 —
            # we want some variety in question styles,
            # not identical questions for similar chunks.
            num_predict=512,
            # Answers should be concise — 512 tokens is plenty
        )

        # Load corpus from BM25 index — it has all our chunks
        self.bm25 = BM25Index()
        if not self.bm25.is_ready():
            raise RuntimeError(
                "BM25 index not built. Run ingestion first: "
                "python -m ingestion.pipeline"
            )

        self.corpus = self.bm25.corpus
        logger.info(f"Corpus loaded: {len(self.corpus)} chunks available")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        num_samples: int = None,
        min_chunk_tokens: int = 80,
    ) -> list[EvalSample]:
        """
        Generate synthetic QA pairs from document chunks.

        Args:
            num_samples:      How many QA pairs to generate.
                              Defaults to settings.eval_sample_size (20)
            min_chunk_tokens: Skip chunks shorter than this.
                              Short chunks rarely contain enough info
                              for a good question.

        Returns:
            List of EvalSample objects with question + ground_truth filled.
            answer and contexts are empty — filled during eval run.
        """
        num_samples = num_samples or settings.eval_sample_size

        # Filter corpus to chunks with enough content
        eligible = [
            chunk for chunk in self.corpus
            if len(chunk.content) // 4 >= min_chunk_tokens
            # reuse our token approximation: chars // 4
            and chunk.content_type.value == "text"
            # Only generate QA from text chunks —
            # tables and image captions are harder to generate good QA from
        ]

        logger.info(
            f"Eligible chunks: {len(eligible)} / {len(self.corpus)} "
            f"(min_tokens={min_chunk_tokens})"
        )

        if len(eligible) < num_samples:
            logger.warning(
                f"Only {len(eligible)} eligible chunks — "
                f"reducing num_samples to {len(eligible)}"
            )
            num_samples = len(eligible)

        # Sample randomly — diverse questions across the document
        sampled = random.sample(eligible, num_samples)
        random.shuffle(sampled)
        # shuffle again to avoid any ordering bias

        logger.info(f"Generating {num_samples} QA pairs...")

        samples = []
        failed = 0

        for i, chunk in enumerate(sampled):
            logger.info(f"  Generating {i+1}/{num_samples} | "
                       f"page={chunk.page_number} section='{chunk.section_title}'")

            sample = self._generate_single(chunk)

            if sample is not None:
                samples.append(sample)
            else:
                failed += 1
                logger.warning(f"  Failed to generate QA for chunk {i+1}")

            # Small delay between LLM calls to avoid overwhelming Ollama
            time.sleep(0.5)

        logger.info(
            f"Generation complete: {len(samples)} success, {failed} failed"
        )
        return samples

    def save(
        self,
        samples: list[EvalSample],
        path: str | Path = "data/eval_testset.json",
    ) -> None:
        """
        Save generated samples to a JSON file.

        The file is human-readable — you can inspect and edit
        questions/answers manually to improve quality.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [s.to_dict() for s in samples]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            # ensure_ascii=False preserves medical unicode chars

        logger.info(f"Saved {len(samples)} samples → {path}")

    def load(
        self,
        path: str | Path = "data/eval_testset.json",
    ) -> list[EvalSample]:
        """
        Load previously generated samples from disk.

        Allows you to generate once, then evaluate multiple times
        without regenerating (which is slow).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Test set not found: {path}. "
                f"Run generator.generate() first."
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for item in data:
            samples.append(EvalSample(
                question=item["question"],
                ground_truth=item["ground_truth"],
                source_chunk={
                    "source_file": item.get("source_file", ""),
                    "page_number": item.get("source_page", 0),
                },
                answer=item.get("answer", ""),
                contexts=item.get("contexts", []),
            ))

        logger.info(f"Loaded {len(samples)} samples from {path}")
        return samples

    # ── Private ───────────────────────────────────────────────────────────────

    def _generate_single(self, chunk) -> Optional[EvalSample]:
        """
        Generate one QA pair from a single chunk.

        Sends the chunk to Ollama and parses the JSON response.
        Returns None if generation or parsing fails.
        """
        prompt = self.GENERATION_PROMPT.format(
            chunk_content=chunk.content[:1000]
            # Limit to 1000 chars — enough for good QA,
            # not so long that Ollama loses focus
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

            # ── Parse JSON from response ──────────────────────
            # LLMs sometimes add markdown code fences around JSON
            # e.g. ```json { ... } ``` — strip those first
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                # Takes content between first ``` pair

            raw = raw.strip()

            parsed = json.loads(raw)
            # json.loads() raises ValueError if the JSON is malformed

            question = parsed.get("question", "").strip()
            answer   = parsed.get("answer", "").strip()

            if not question or not answer:
                logger.warning("Empty question or answer in response")
                return None

            if len(question) < 10:
                # Sanity check — real questions are at least 10 chars
                logger.warning(f"Question too short: '{question}'")
                return None

            return EvalSample(
                question=question,
                ground_truth=answer,
                source_chunk=chunk.to_dict(),
            )

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e} | Raw: '{raw[:100]}'")
            return None
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return None