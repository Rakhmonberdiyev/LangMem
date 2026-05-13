"""Unified context ingestion: merges session history + LTM facts into messages."""

from datetime import datetime

def _system_prompt() -> str:
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    return f"""\
You are a helpful AI assistant with long-term memory and access to a personal knowledge base.
Current date: {date_str}  |  Current time: {time_str} (UTC+5 Tashkent)

## Tools you have
- **rag_search**: Search documents and files the user has previously uploaded (PDFs, text files, CVs, reports, etc.).
  → Use this whenever the user asks about a file they sent, references a document, or asks you to recall/summarize uploaded content.
- **web_search**: Search the internet for real-time or factual information you don't already know.

## Rules
- If the user asks about a document, file, or anything they previously uploaded → ALWAYS call rag_search first.
- Never say you cannot access a file — if a file was uploaded it is in the knowledge base; search for it.
- Combine rag_search results with your reasoning to give accurate, grounded answers.\
"""


def build_messages(
    user_input: str,
    session_history: list[dict],
    ltm_facts: str,
    user_docs: list[str] | None = None,
) -> list[dict]:
    """
    Returns the full messages array ready to send to the LLM:
      [system] → [recent session history] → [user]
    LTM facts, current date/time, and uploaded document list are injected into the system message.
    """
    sys_content = _system_prompt()   # fresh timestamp on every call

    if user_docs:
        doc_list = "\n".join(f"  - {d}" for d in user_docs)
        sys_content += (
            f"\n\n[Documents uploaded by this user — searchable via rag_search, most recent first]:\n"
            f"{doc_list}\n"
            f"When the user says 'this file', 'this document', 'last pdf', or similar, "
            f"they mean the most recent one: '{user_docs[0]}'."
        )

    if ltm_facts:
        sys_content += f"\n\n[Long-term memory about this user]:\n{ltm_facts}"

    messages: list[dict] = [{"role": "system", "content": sys_content}]
    messages.extend(session_history[-20:])
    messages.append({"role": "user", "content": user_input})
    return messages
