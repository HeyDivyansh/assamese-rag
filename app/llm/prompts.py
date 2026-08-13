"""Bilingual prompt construction for sarvam-105b."""
from __future__ import annotations

from app.cleaner.unicode import detect_query_language
from app.retrieval.types import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers strictly from the provided "
    "document context (Assamese and/or English). Follow these rules:\n"
    "1. Reply ONLY with the final answer — no analysis, reasoning, or step-by-step "
    "explanation.\n"
    "2. Answer in the SAME language as the user's question. If they write in "
    "Assamese (অসমীয়া), reply in Assamese. If they write in English, reply in "
    "English. For mixed-language questions, use the dominant language.\n"
    "3. Use ONLY the provided context. If the answer is not in the context, reply "
    "briefly that the information was not found in the documents (in the user's "
    "language).\n"
    "4. Cite sources you used with [S#] markers.\n"
    "5. Be concise — one or two short sentences when possible."
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


def _build_user_turn(query: str, context: str, query_language: str) -> str:
    if query_language == "en":
        return (
            f"Context:\n{context}\n\n"
            "----\n"
            f"Question: {query}\n\n"
            "Answer from the context above only. Cite sources with [S#] markers."
        )
    return (
        "প্ৰসংগ (Context):\n"
        f"{context}\n\n"
        "----\n"
        f"প্ৰশ্ন (Question): {query}\n\n"
        "ওপৰৰ প্ৰসংগৰ ভিত্তিত উত্তৰ দিয়ক আৰু ব্যৱহৃত উৎসবোৰ [S#] "
        "চিহ্নেৰে উল্লেখ কৰক।"
    )


def build_messages(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
    *,
    query_language: str | None = None,
) -> list[dict]:
    """Assemble the chat messages list: system + memory + context+query."""
    lang = query_language or detect_query_language(query)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history)

    context = build_context_block(chunks)
    messages.append({"role": "user", "content": _build_user_turn(query, context, lang)})
    return messages
