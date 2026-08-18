"""Bilingual prompt construction for sarvam-105b."""
from __future__ import annotations

from app.cleaner.unicode import detect_query_language
from app.retrieval.types import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a helpful multilingual voice assistant.\n"
    "1. Answer naturally, conversationally, and concisely.\n"
    "2. ALWAYS answer in the SAME language as the user's question.\n"
    "3. The supported languages for this voice assistant are English and Kannada.\n"
    "4. If the user's question is in English, answer in English.\n"
    "5. If the user's question is in Kannada, answer in Kannada.\n"
    "6. NEVER translate the user's question into another language before answering.\n"
    "7. NEVER default to Assamese.\n"
    "8. For casual conversation, greetings, small talk, and general questions "
    "that do not require documents, answer directly using your general knowledge.\n"
    "9. Keep voice answers short and conversational. Normally answer in 1-3 sentences.\n"
    "10. If the user asks for an explanation, give a concise explanation using short "
    "sentences or a few brief points. Do not produce long paragraphs unless the user "
    "explicitly asks for a detailed answer.\n"
    "11. Do not provide analysis or reasoning unless requested.\n"
    "12. Do not use [S#] citations for casual conversation. For document-based "
    "answers, cite relevant sources with [S#] markers."
)

def build_context_block(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        text = c.payload.get("expanded_text") or c.text
        title = c.section_title or ""
        page = f"p.{c.page_number}" if c.page_number is not None else ""
        header = f"[S{i}]" + (f" {title}" if title else "") + (f" ({page})" if page else "")
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def _build_user_turn(
    query: str,
    context: str,
    query_language: str,
) -> str:
    return (
        f"Context:\n{context}\n\n"
        "----\n"
        f"User question: {query}\n\n"
        "Answer the user's question appropriately. "
        "If relevant document context is provided, use it to answer accurately "
        "and cite the relevant sources with [S#] markers. "
        "Answer in the detected user language."
    )


def build_messages(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
    *,
    query_language: str | None = None,
) -> list[dict]:
    """Assemble chat messages using the STT-detected language."""
    lang = query_language or detect_query_language(query)

    language_name = {
        "en-IN": "English",
        "hi-IN": "Hindi",
        "kn-IN": "Kannada",
        "as-IN": "Assamese",
        "ta-IN": "Tamil",
        "te-IN": "Telugu",
    }.get(lang, "the same language as the user")

    system_prompt = (
        SYSTEM_PROMPT
        + f"\n\nIMPORTANT: The detected user language is {language_name}. "
        f"You MUST answer in {language_name}. "
        "Do not translate the answer into Assamese unless Assamese is the detected language."
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt}
    ]

    if history:
        messages.extend(history)

    context = build_context_block(chunks)

    messages.append(
        {
            "role": "user",
            "content": _build_user_turn(query, context, lang),
        }
    )

    return messages
