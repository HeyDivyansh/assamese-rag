"""Assamese-aware prompt construction for sarvam-30b."""
from __future__ import annotations

from app.retrieval.types import RetrievedChunk

SYSTEM_PROMPT = (
    "আপুনি এজন সহায়ক যিয়ে কেৱল প্ৰদান কৰা নথিৰ প্ৰসংগৰ ওপৰত ভিত্তি কৰি উত্তৰ দিয়ে।\n"
    "You are a helpful assistant that answers strictly from the provided "
    "Assamese document context. Follow these rules:\n"
    "1. Reply ONLY with the final answer — no analysis, reasoning, or step-by-step "
    "explanation.\n"
    "2. Answer in Assamese (অসমীয়া). If the user asks in English, still answer in "
    "Assamese unless they explicitly ask for another language.\n"
    "3. Use ONLY the provided context. If the answer is not in the context, reply "
    "briefly in Assamese that the information was not found in the documents.\n"
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


def build_messages(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> list[dict]:
    """Assemble the chat messages list: system + memory + context+query."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history)

    context = build_context_block(chunks)
    user_turn = (
        "প্ৰসংগ (Context):\n"
        f"{context}\n\n"
        "----\n"
        f"প্ৰশ্ন (Question): {query}\n\n"
        "ওপৰৰ প্ৰসংগৰ ভিত্তিত উত্তৰ দিয়ক আৰু ব্যৱহৃত উৎসবোৰ [S#] "
        "চিহ্নেৰে উল্লেখ কৰক।"
    )
    messages.append({"role": "user", "content": user_turn})
    return messages
