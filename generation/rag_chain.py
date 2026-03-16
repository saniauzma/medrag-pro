# generation/rag_chain.py
# -----------------------
# The RAG chain — wires together the full query pipeline:
#
#   User question
#       ↓
#   HybridRetriever    → top-10 candidates (dense + BM25 + RRF)
#       ↓
#   MedReranker        → top-5 reranked chunks (CrossEncoder)
#       ↓
#   format_context()   → structured context string
#       ↓
#   MEDICAL_RAG_PROMPT → filled prompt
#       ↓
#   Ollama (llama3.2:3b) → answer with citations
#       ↓
#   RAGResponse        → answer + sources + metadata

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from langchain_ollama import ChatOllama
# LangChain's Ollama integration.
# ChatOllama wraps the Ollama REST API as a LangChain chat model.
# Supports streaming, system prompts, and temperature control.

from langchain_core.messages import HumanMessage, SystemMessage
# LangChain message types for chat models.
# SystemMessage → sets the model's behavior/persona
# HumanMessage  → the user's actual question (with context injected)

from config import settings
from generation.prompt_templates import (
    MEDICAL_RAG_PROMPT,
    STANDALONE_QUESTION_PROMPT,
    NO_CONTEXT_RESPONSE,
    format_context,
)
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import MedReranker

logger = logging.getLogger(__name__)


# ─── Response Model ───────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    Structured response from the RAG chain.

    Contains not just the answer but full provenance —
    which chunks were retrieved, how they were ranked,
    and how long each step took.

    This is what the FastAPI endpoint returns to the UI.
    """
    answer: str
    # The LLM's answer with inline citations

    sources: list[dict]
    # The chunks used to generate the answer.
    # Each dict has: content, source_file, page_number,
    #                section_title, rerank_score, rrf_score

    query: str
    # The original user question (before any rewriting)

    standalone_query: str = ""
    # The rewritten standalone question (for multi-turn)
    # Same as query if no rewriting was needed

    retrieval_time: float = 0.0
    # Seconds taken by hybrid retrieval + reranking

    generation_time: float = 0.0
    # Seconds taken by Ollama LLM generation

    total_time: float = 0.0
    # End-to-end latency

    num_chunks_retrieved: int = 0
    # How many chunks came back from hybrid search

    num_chunks_used: int = 0
    # How many chunks were passed to the LLM (after reranking)

    def to_dict(self) -> dict:
        """Serialize for FastAPI JSON response."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "standalone_query": self.standalone_query,
            "timing": {
                "retrieval_seconds": round(self.retrieval_time, 2),
                "generation_seconds": round(self.generation_time, 2),
                "total_seconds": round(self.total_time, 2),
            },
            "stats": {
                "chunks_retrieved": self.num_chunks_retrieved,
                "chunks_used": self.num_chunks_used,
            },
        }


# ─── RAG Chain ────────────────────────────────────────────────────────────────

class MedRAGChain:
    """
    Full RAG chain for medical question answering.

    Orchestrates retrieval → reranking → generation into
    a single .query() call.

    Usage:
        chain = MedRAGChain()
        response = chain.query("What are the side effects of metformin?")
        print(response.answer)
        print(response.sources)
    """

    def __init__(self):
        # ── Retrieval components ──────────────────────────────
        self.retriever = HybridRetriever()
        self.reranker  = MedReranker()

        # ── LLM (Ollama) ──────────────────────────────────────
        logger.info(f"Loading LLM: {settings.llm_model} via Ollama")
        self.llm = ChatOllama(
            model=settings.llm_model,
            # "llama3.2:3b" — must be pulled in Ollama already

            base_url=settings.ollama_base_url,
            # "http://localhost:11434"

            temperature=settings.llm_temperature,
            # 0.0 — fully deterministic for medical Q&A.
            # Same question always gets same answer.
            # Critical for reproducibility in clinical settings.

            num_predict=settings.llm_max_tokens,
            # Max tokens to generate in response.
            # 1024 is enough for detailed medical answers.
            num_gpu=99,
        )
        logger.info("LLM ready ✅")

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        source_file: str = None,
        chat_history: list[tuple[str, str]] = None,
        num_candidates: int = 10,
    ) -> RAGResponse:
        """
        Answer a medical question using RAG.

        Args:
            question:       The user's question in plain text
            source_file:    Optional — restrict to one PDF file
            chat_history:   Optional — list of (question, answer) tuples
                            for multi-turn conversation support
            num_candidates: How many chunks to retrieve before reranking
                            More = better recall, slower reranking

        Returns:
            RAGResponse with answer, sources, and timing metadata
        """
        start_time = time.time()
        logger.info(f"Query: '{question[:80]}'")

        # ── Step 1: Rewrite question if multi-turn ────────────
        standalone = self._make_standalone(question, chat_history)
        # If no chat history, standalone == question (no change)
        # If follow-up question, standalone is rewritten to be self-contained

        # ── Step 2: Retrieve ──────────────────────────────────
        retrieval_start = time.time()

        candidates = self.retriever.retrieve(
            query=standalone,
            k=num_candidates,
            source_file=source_file,
        )
        # Returns top-N chunks from hybrid search (dense + BM25 + RRF)

        if not candidates:
            # No chunks found — return early with honest response
            logger.warning("No candidates retrieved")
            return RAGResponse(
                answer=NO_CONTEXT_RESPONSE,
                sources=[],
                query=question,
                standalone_query=standalone,
                total_time=round(time.time() - start_time, 2),
            )

        # ── Step 3: Rerank ────────────────────────────────────
        reranked = self.reranker.rerank(
            query=standalone,
            candidates=candidates,
            top_k=settings.rerank_top_k,
            # 5 chunks go to the LLM — enough context, not too noisy
        )

        retrieval_time = time.time() - retrieval_start
        logger.info(
            f"Retrieval + reranking: {retrieval_time:.2f}s | "
            f"{len(candidates)} candidates → {len(reranked)} chunks"
        )

        # ── Step 4: Format context ────────────────────────────
        context = format_context(reranked)
        # Converts chunk dicts into numbered [SOURCE N] blocks

        # ── Step 5: Build prompt ──────────────────────────────
        prompt_text = MEDICAL_RAG_PROMPT.format(
            context=context,
            question=standalone,
        )
        # Fills in {context} and {question} in the template

        # ── Step 6: Generate answer ───────────────────────────
        generation_start = time.time()

        messages = [
            SystemMessage(content=(
                "You are a precise medical research assistant. "
                "Answer questions strictly based on provided sources. "
                "Always cite sources. Never hallucinate."
            )),
            # SystemMessage sets the model's overall behavior.
            # Reinforces the grounding instruction from the prompt.

            HumanMessage(content=prompt_text),
            # The full prompt with context + question goes here.
            # We use HumanMessage not SystemMessage because
            # the context and rules are part of the user's request.
        ]

        logger.info("Generating answer...")
        response = self.llm.invoke(messages)
        # .invoke() sends the messages to Ollama and waits for the full response.
        # Returns an AIMessage object — the answer is in response.content

        generation_time = time.time() - generation_start
        total_time = time.time() - start_time

        logger.info(f"Generation: {generation_time:.2f}s | Total: {total_time:.2f}s")

        # ── Step 7: Build response ────────────────────────────
        return RAGResponse(
            answer=response.content,
            sources=reranked,
            query=question,
            standalone_query=standalone,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            num_chunks_retrieved=len(candidates),
            num_chunks_used=len(reranked),
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _make_standalone(
        self,
        question: str,
        chat_history: list[tuple[str, str]] | None,
    ) -> str:
        """
        Rewrite a follow-up question into a standalone question.

        Only activates when chat_history is non-empty.
        Uses STANDALONE_QUESTION_PROMPT + Ollama to rewrite.

        Example:
            history:  [("What is metformin?", "Metformin is a biguanide...")]
            question: "What are its side effects?"
            result:   "What are the side effects of metformin?"

        Without this:
            "its side effects" → retrieves nothing useful
        With this:
            "side effects of metformin" → retrieves relevant chunks
        """
        if not chat_history:
            # No history → question is already standalone
            return question

        # Format history as a readable string
        history_str = "\n".join(
            f"Human: {q}\nAssistant: {a}"
            for q, a in chat_history[-3:]
            # Only use last 3 turns — older history dilutes the rewrite
        )

        prompt = STANDALONE_QUESTION_PROMPT.format(
            chat_history=history_str,
            question=question,
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            standalone = response.content.strip()
            if standalone and standalone != question:
                logger.info(f"Rewritten: '{question}' → '{standalone}'")
            return standalone
        except Exception as e:
            logger.warning(f"Question rewriting failed: {e}. Using original.")
            return question