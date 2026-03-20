from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, List

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import OPENROUTER_API_KEY, MODEL_NAME

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# ── Data Classes ───────────────────────────────────

@dataclass
class Chunk:
    text: str
    source: str
    index: int


@dataclass
class RetrievedContext:
    text: str
    source: str
    score: float


# ── RAG Pipeline ───────────────────────────────────

class RAGPipeline:

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        openrouter_api_key: str | None = None,
        model_name: str | None = None,
        embed_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ):
        self.api_key = openrouter_api_key or OPENROUTER_API_KEY
        self.model_name = model_name or MODEL_NAME
        self.top_k = top_k

        # Embedding model
        self.embedder = SentenceTransformer(embed_model)

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Storage
        self.chunks: List[Chunk] = []
        self.index = None
        self.embed_dim = 0

    # ── Build Index ─────────────────────────────────

    def build_index(self, uploaded_files: List[Any]) -> dict:
        all_chunks = []

        for uf in uploaded_files:
            text = self._extract_text(uf)

            if not text.strip():
                continue

            splits = self.splitter.split_text(text)

            for i, chunk in enumerate(splits):
                all_chunks.append(Chunk(chunk, uf.filename if hasattr(uf, "filename") else uf.name, i))

        if not all_chunks:
            raise ValueError("No valid text found in uploaded files.")

        self.chunks = all_chunks

        texts = [c.text for c in all_chunks]

        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        embeddings = np.array(embeddings, dtype="float32")

        self.embed_dim = embeddings.shape[1]

        # FAISS Index
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(embeddings)

        return {
            "n_docs": len(uploaded_files),
            "n_chunks": len(all_chunks),
            "embed_dim": self.embed_dim,
        }

    # ── Query ───────────────────────────────────────

    def query(self, question: str, top_k: int | None = None) -> dict:

        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        k = min(top_k or self.top_k, len(self.chunks))

        # Embed query
        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        scores, indices = self.index.search(q_emb, k)

        contexts = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            chunk = self.chunks[idx]
            contexts.append(
                RetrievedContext(
                    text=chunk.text,
                    source=chunk.source,
                    score=float(score),
                )
            )

        answer = self._generate(question, contexts)

        return {
            "answer": answer,
            "contexts": [
                {"text": c.text, "source": c.source, "score": c.score}
                for c in contexts
            ],
        }

    # ── LLM Generation ──────────────────────────────

    def _generate(self, question: str, contexts: List[RetrievedContext]) -> str:

        if not self.api_key:
            return "⚠️ Missing OpenRouter API key."

        context_block = "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c in contexts
        )

        system_prompt = (
            "You are a helpful assistant. Answer ONLY using the provided context. "
            "If the answer is not found, say you don't know."
        )

        user_prompt = f"""
Context:
{context_block}

Question: {question}
Answer:
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "RAG App",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        }

        try:
            resp = requests.post(
                self.OPENROUTER_URL,
                json=payload,
                headers=headers,
                timeout=60,
            )

            if resp.status_code != 200:
                return f"❌ API Error: {resp.text}"

            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            return f"❌ Request failed: {str(e)}"

    # ── Text Extraction ─────────────────────────────

    def _extract_text(self, uploaded_file: Any) -> str:

        # Reset pointer (important for Streamlit)
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        name = (
            uploaded_file.filename.lower()
            if hasattr(uploaded_file, "filename")
            else uploaded_file.name.lower()
        )

        raw = (
            uploaded_file.file.read()
            if hasattr(uploaded_file, "file")
            else uploaded_file.read()
        )

        # PDF
        if name.endswith(".pdf"):
            if not PDF_SUPPORT:
                raise ImportError("Install pdfplumber for PDF support.")

            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
            return "\n\n".join(pages)

        # Text
        try:
            return raw.decode("utf-8")
        except:
            return raw.decode("latin-1")