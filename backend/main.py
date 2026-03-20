from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import uvicorn

from .rag_pipeline import RAGPipeline
from .config import OPENROUTER_API_KEY, MODEL_NAME

app = FastAPI(
    title="RAG API",
    description="Production-ready RAG backend",
    version="1.0"
)

# Global pipeline instance
pipeline = None


# ── Health check ─────────────────────────────────────
@app.get("/")
def root():
    return {"message": "RAG API is running 🚀"}


# ── Build Index ─────────────────────────────────────
@app.post("/build")
async def build_index(files: List[UploadFile] = File(...)):
    global pipeline

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        # Initialize pipeline
        pipeline = RAGPipeline(
            openrouter_api_key=OPENROUTER_API_KEY,
            model_name=MODEL_NAME
        )

        # Build index
        stats = pipeline.build_index(files)

        return {
            "status": "success",
            "message": "Index built successfully",
            "stats": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Query ───────────────────────────────────────────
@app.get("/query")
def query(q: str):
    global pipeline

    if not pipeline:
        raise HTTPException(
            status_code=400,
            detail="Index not built. Call /build first."
        )

    if not q.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    try:
        result = pipeline.query(q)

        return {
            "status": "success",
            "answer": result["answer"],
            "contexts": result["contexts"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run Server ──────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
