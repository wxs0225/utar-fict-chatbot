"""
UTAR FICT Chatbot - RAG Core Engine
Uses Groq LLM (FREE) + Retrieval-Augmented Generation (RAG)
"""

import re
from groq import Groq

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
TOP_K         = 5
MODEL         = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are a helpful academic assistant for UTAR (Universiti Tunku Abdul Rahman)
FICT (Faculty of Information & Communication Technology) students.
You answer questions about the Final Year Project (FYP) guidelines, procedures,
handbook rules, FAQs, and other faculty/university policies.

Answer ONLY based on the provided context. If the answer is not in the context,
say "I'm sorry, I don't have that information in my knowledge base. Please visit
www.utar.edu.my or contact your faculty office directly."

Be concise, friendly, and accurate. Use bullet points where helpful."""


# ── Simple In-Memory Vector Store ─────────────────────────────────────────────
class SimpleVectorStore:
    def __init__(self):
        self.chunks: list[dict] = []

    def add_document(self, text: str, source: str):
        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        start = 0
        while start < len(text):
            chunk_text = text[start: start + CHUNK_SIZE].strip()
            if len(chunk_text) > 50:
                self.chunks.append({
                    "text":   chunk_text,
                    "source": source,
                    "tokens": self._tokenize(chunk_text),
                })
            start += step

    def _tokenize(self, text: str) -> set[str]:
        stops = {
            "the","a","an","is","are","was","were","be","been","being",
            "have","has","had","do","does","did","will","would","could",
            "should","may","might","shall","to","of","in","for","on",
            "with","at","by","from","as","or","and","but","not","it",
            "its","this","that","these","those","he","she","they","we",
            "you","i","my","your","our","their","what","how","when",
            "where","which","who","if","then","than","so","can","also",
        }
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return {w for w in words if w not in stops and len(w) > 2}

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        if not self.chunks:
            return []
        q_tokens = self._tokenize(query)
        scored = []
        for chunk in self.chunks:
            overlap = len(q_tokens & chunk["tokens"])
            if overlap:
                union = len(q_tokens | chunk["tokens"])
                scored.append((overlap / union, chunk))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [c for _, c in scored[:top_k]]


# ── RAG Pipeline ───────────────────────────────────────────────────────────────
def build_context(chunks: list[dict]) -> str:
    parts = [f"[Source: {c['source']}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


def ask(
    query: str,
    store: SimpleVectorStore,
    client: Groq,
    history: list[dict],
) -> str:
    chunks  = store.search(query, top_k=TOP_K)
    context = build_context(chunks) if chunks else "No relevant documents found."

    user_msg = (
        f"Use the following knowledge base excerpts to answer the student's question.\n\n"
        f"=== KNOWLEDGE BASE ===\n{context}\n=== END ===\n\n"
        f"Student's question: {query}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-10:]
    messages.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
    )
    answer = response.choices[0].message.content

    history.append({"role": "user",      "content": query})
    history.append({"role": "assistant", "content": answer})
    if len(history) > 20:
        history[:] = history[-20:]

    return answer