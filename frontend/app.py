import streamlit as st
import requests
import time

# 🔗 Backend URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("🏗️ Construction AI Assistant")

# ── Session State ─────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False


# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF / TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    build_btn = st.button("🔨 Build Index")

    st.divider()
    st.caption("⚠️ Make sure backend is running on port 8000")


# ── Build Index ───────────────────────────────
if build_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload files first")
    else:
        with st.spinner("Building index..."):

            try:
                files = [
                    ("files", (f.name, f, f.type))
                    for f in uploaded_files
                ]

                res = requests.post(f"{API_URL}/build", files=files)

                if res.status_code == 200:
                    st.session_state.index_ready = True
                    st.sidebar.success("✅ Index built successfully")
                else:
                    st.sidebar.error(res.text)

            except Exception as e:
                st.sidebar.error(f"❌ Backend error: {e}")


# ── Chat History ──────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ── Chat UI ───────────────────────────────────
if st.session_state.index_ready:

    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:

        # User message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving + 🤖 Generating..."):

                try:
                    start = time.time()

                    res = requests.get(
                        f"{API_URL}/query",
                        params={"q": user_input}
                    )

                    if res.status_code != 200:
                        st.error(res.text)
                        answer = "❌ Backend error"
                        contexts = []
                    else:
                        data = res.json()
                        answer = data.get("answer", "")
                        contexts = data.get("contexts", [])

                    latency = round(time.time() - start, 2)

                    # Show answer
                    st.write(answer)
                    st.caption(f"⏱ Response time: {latency}s")

                    # Show retrieved context
                    with st.expander(f"📚 Retrieved Context ({len(contexts)} chunks)"):
                        for i, ctx in enumerate(contexts, 1):
                            st.markdown(f"""
**Chunk {i}**  
Score: {ctx['score']:.3f}  
Source: {ctx['source']}

{ctx['text']}
---
""")

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err
                    })

else:
    st.info("👈 Upload documents and build index to start chatting")