"""
UTAR FICT Chatbot - Streamlit Web Interface
Auto-loads from data/ folder on startup
Supports both local and Streamlit Cloud deployment
Run with: streamlit run app.py
"""

import os
import io
import csv
import tempfile
from pathlib import Path
import streamlit as st
from chatbot import ask, SimpleVectorStore
from groq import Groq, AuthenticationError

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UTAR FICT Chatbot",
    page_icon="🎓",
    layout="centered",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .title-box {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .pill-green {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .pill-orange {
        display: inline-block;
        background: #fff3e0;
        color: #e65100;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h2 style="margin:0">🎓 UTAR FICT Student Chatbot</h2>
    <p style="margin:4px 0 0; opacity:0.85; font-size:0.9rem">
        Powered by Groq LLM (Free) + RAG &nbsp;·&nbsp;
        Ask anything about FYP, handbook &amp; FAQs
    </p>
</div>
""", unsafe_allow_html=True)


# ── Get API Key (Streamlit secrets or local env) ───────────────────────────────
def get_api_key() -> str:
    # Try Streamlit secrets first (for deployment)
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    # Then try environment variable (for local)
    return os.environ.get("GROQ_API_KEY", "")


# ── File Loaders ───────────────────────────────────────────────────────────────
def load_pdf(data: bytes, name: str = "") -> str:
    try:
        from pdfminer.high_level import extract_text
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        text = extract_text(tmp_path)
        os.unlink(tmp_path)
        return text or ""
    except ImportError:
        st.error("pdfminer.six missing — run: pip install pdfminer.six")
        return ""
    except Exception as e:
        st.error(f"PDF read error ({name}): {e}")
        return ""

def load_csv(data: bytes, name: str = "") -> str:
    try:
        content = data.decode("utf-8-sig")
        reader  = csv.DictReader(io.StringIO(content))
        lines   = []
        for row in reader:
            pairs = [f"{k}: {v}" for k, v in row.items() if v and v.strip()]
            if pairs:
                lines.append("  |  ".join(pairs))
        return "\n".join(lines)
    except Exception as e:
        st.error(f"CSV read error ({name}): {e}")
        return ""

def load_txt(data: bytes, name: str = "") -> str:
    try:
        return data.decode("utf-8")
    except Exception as e:
        st.error(f"TXT read error ({name}): {e}")
        return ""

LOADERS = {"pdf": load_pdf, "csv": load_csv, "txt": load_txt}


# ── Auto-load from data/ folder ────────────────────────────────────────────────
def build_store_from_folder(data_dir: Path) -> SimpleVectorStore:
    store = SimpleVectorStore()
    if not data_dir.exists():
        return store
    for path in sorted(data_dir.iterdir()):
        ext    = path.suffix.lower().lstrip(".")
        loader = LOADERS.get(ext)
        if loader:
            data   = path.read_bytes()
            text   = loader(data, name=path.name)
            if text.strip():
                store.add_document(text, source=path.name)
    return store

def add_uploads_to_store(store: SimpleVectorStore, uploaded_files) -> SimpleVectorStore:
    for uf in uploaded_files:
        ext    = uf.name.rsplit(".", 1)[-1].lower()
        loader = LOADERS.get(ext)
        if loader:
            text = loader(uf.read(), name=uf.name)
            if text.strip():
                store.add_document(text, source=uf.name)
    return store


# ── API Key ────────────────────────────────────────────────────────────────────
auto_api_key = get_api_key()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Hide API key input if already loaded from secrets
    if auto_api_key:
        api_key = auto_api_key
        st.success("✓ Ready to chat!")
    else:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Free at console.groq.com",
        )
        st.caption("🆓 Groq is 100% free")

    st.divider()

    st.subheader("📂 Knowledge Base Files")
    data_dir = Path("data")
    if data_dir.exists():
        auto_files = [
            f for f in data_dir.iterdir()
            if f.suffix.lower() in {".pdf", ".csv", ".txt"}
        ]
        if auto_files:
            for f in auto_files:
                st.markdown(f"✅ `{f.name}`")
        else:
            st.warning("No files in `data/` folder.")
    else:
        st.warning("`data/` folder not found.")

    if st.button("🔄 Reload Knowledge Base", use_container_width=True):
        if "store" in st.session_state:
            del st.session_state["store"]
        st.rerun()

    st.divider()

    st.subheader("➕ Upload Extra Files")
    uploaded_files = st.file_uploader(
        "Add more PDF/CSV/TXT",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        if st.button("➕ Add to Knowledge Base", use_container_width=True):
            store = st.session_state.get("store", SimpleVectorStore())
            store = add_uploads_to_store(store, uploaded_files)
            st.session_state.store = store
            st.success(f"✓ Added! Total: {len(store.chunks)} chunks")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()

    st.caption("UTAR FICT Perak Campus · Chatbot v3.0")


# ── Auto-load knowledge base ───────────────────────────────────────────────────
if "store" not in st.session_state:
    with st.spinner("📚 Loading knowledge base..."):
        st.session_state.store    = build_store_from_folder(Path("data"))
        st.session_state.messages = []
        st.session_state.history  = []

store = st.session_state.store
n     = len(store.chunks)

if n:
    st.markdown(
        f'<div class="pill-green">✓ {n} chunks indexed — knowledge base ready!</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="pill-orange">⚠ No knowledge base loaded. Put files in <code>data/</code> folder.</div>',
        unsafe_allow_html=True
    )

# ── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history  = []

# ── Starter Questions ──────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    starters = [
        "What are the FYP submission deadlines?",
        "What is the word count for Project II?",
        "How do I write a literature review?",
        "What is the report binding format?",
        "How often should I meet my supervisor?",
        "What happens if I plagiarise?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(starters):
        if cols[i % 2].button(q, key=f"s{i}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

# ── Render Chat History ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Chat Input ─────────────────────────────────────────────────────────────────
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")
else:
    query = st.chat_input("Ask about FYP, handbook rules, procedures…")

# ── Generate Answer ────────────────────────────────────────────────────────────
if query:
    if not api_key:
        st.error("Enter your Groq API key in the sidebar first.")
        st.stop()
    if not n:
        st.error("Knowledge base is empty. Put your files in the data/ folder.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                client = Groq(api_key=api_key)
                answer = ask(query, store, client, st.session_state.history)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except AuthenticationError:
                st.error("Invalid Groq API key. Check it at console.groq.com")
            except Exception as e:
                st.error(f"Error: {e}")
