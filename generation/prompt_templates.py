# generation/prompt_templates.py
# --------------------------------
# Prompt templates for medical RAG.
#
# Why separate prompts from the chain?
#   - Easy to iterate on prompts without touching chain logic
#   - Easy to A/B test different prompt versions
#   - Easy to add new prompt types (summary, comparison, etc.)
#   - Prompts are the most important tuning lever in RAG

from __future__ import annotations

from langchain_core.prompts import PromptTemplate
# PromptTemplate is LangChain's string template with named variables.
# Variables are written as {variable_name} in the template string.
# PromptTemplate.format(variable_name="value") fills them in.


# ─── Context Formatter ────────────────────────────────────────────────────────

def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a structured context string
    that gets injected into the prompt.

    Each chunk is formatted as a numbered source block with metadata.
    This structure helps the LLM:
      1. Distinguish between different sources
      2. Cite specific sources in its answer
      3. Understand the document structure (page, section)

    Example output:
        [SOURCE 1]
        File: diabetes_study.pdf | Page: 4 | Section: Results
        Content: Metformin 500mg twice daily was associated with...

        [SOURCE 2]
        File: clinical_trial.pdf | Page: 2 | Section: Methods
        Content: Patients were randomized to receive...

    Args:
        chunks: List of chunk dicts from MedReranker.rerank()
                Each must have: content, source_file, page_number, section_title

    Returns:
        Formatted string ready to inject into prompt
    """
    if not chunks:
        return "No relevant context found."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        # Build a clean source header for each chunk
        source_file    = chunk.get("source_file", "unknown")
        page_number    = chunk.get("page_number", "?")
        section_title  = chunk.get("section_title", "")
        content_type   = chunk.get("content_type", "text")

        # Section line — only include if non-empty
        section_str = f" | Section: {section_title}" if section_title else ""

        # Content type indicator — helps LLM interpret tables/figures
        type_str = ""
        if content_type == "table":
            type_str = " [TABLE]"
        elif content_type == "image":
            type_str = " [FIGURE]"

        parts.append(
            f"[SOURCE {i}]{type_str}\n"
            f"File: {source_file} | Page: {page_number}{section_str}\n"
            f"Content: {chunk['content']}"
        )

    return "\n\n".join(parts)
    # Double newline separates each source block clearly


# ─── Main RAG Prompt ──────────────────────────────────────────────────────────

MEDICAL_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    # These two variables get filled in by the chain at query time.
    # "context" → formatted chunks from format_context()
    # "question" → the user's raw question

    template="""You are a precise medical research assistant.
Your role is to answer questions based strictly on the provided source documents.

CRITICAL RULES:
1. Answer ONLY using information from the provided sources below.
2. If the answer is not in the sources, respond exactly with:
   "The provided documents do not contain sufficient information to answer this question."
3. Never use your general knowledge to fill gaps — only use the sources.
4. Cite your sources after each claim using: [Source N, Page X]
5. For numerical values (dosages, statistics, p-values), quote them exactly as written.
6. If sources contradict each other, point out the contradiction explicitly.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
Provide a clear, structured answer using only the above sources.
Include citations [Source N, Page X] for every factual claim.""",
)
# Why this prompt works well for medical RAG:
#
# "ONLY using information from sources" — prevents hallucination
# "does not contain sufficient information" — exact phrase to detect uncertainty
# "Never use general knowledge" — explicitly blocks parametric knowledge
# "quote exactly as written" — critical for drug dosages and statistics
# "point out contradictions" — handles conflicting evidence honestly


# ─── Standalone Question Prompt ───────────────────────────────────────────────
# Used to rewrite conversational follow-up questions into standalone queries.
# Example:
#   History: "What is metformin?"
#   Follow-up: "What are its side effects?"
#   Standalone: "What are the side effects of metformin?"
#
# Why this matters:
#   "its side effects" can't be retrieved — "metformin side effects" can.
#   Without this step, multi-turn conversations degrade retrieval quality.

STANDALONE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Given the following conversation history and a follow-up question,
rewrite the follow-up question as a complete standalone question that can be
understood without the conversation history.

IMPORTANT:
- Keep all specific medical terms, drug names, and numbers from the original question
- If the question is already standalone, return it unchanged
- Return ONLY the rewritten question, nothing else

CONVERSATION HISTORY:
{chat_history}

FOLLOW-UP QUESTION:
{question}

STANDALONE QUESTION:""",
)


# ─── Summary Prompt ───────────────────────────────────────────────────────────
# Used to summarize a retrieved document or set of chunks.

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context", "focus"],
    template="""You are a medical research assistant. Summarize the following
source documents with a focus on: {focus}

RULES:
1. Only include information present in the sources
2. Organize by topic, not by source order
3. Preserve all numerical values exactly (dosages, statistics, dates)
4. Flag any contradictions between sources
5. End with: "Sources: [list of source files used]"

SOURCES:
{context}

SUMMARY:""",
)


# ─── No-Context Fallback ──────────────────────────────────────────────────────
# Used when retrieval returns empty results.
# Better to be honest than to hallucinate.

NO_CONTEXT_RESPONSE = (
    "I was unable to find relevant information in the provided documents "
    "to answer your question. Please try:\n"
    "1. Rephrasing your question with different terminology\n"
    "2. Uploading additional relevant documents\n"
    "3. Checking if your question is within the scope of the indexed documents"
)