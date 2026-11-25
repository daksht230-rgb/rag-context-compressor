from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from compressor_engine import RAGContextCompressor

app = FastAPI(
    title="RAG Context Compressor API",
    description="High-speed BM25 sentence-level context pruning service to eliminate RAG noise and reduce LLM prompt token costs.",
    version="1.0.0",
)

compressor = RAGContextCompressor()


class CompressRequest(BaseModel):
    query: str = Field(..., min_length=2, example="What is the API rate limit for enterprise workspaces?")
    chunks: List[str] = Field(
        ...,
        min_length=1,
        example=[
            "Welcome to our developer portal. We updated our UI theme last month. Enterprise workspaces allow up to 1,200 requests per minute with burst protection.",
            "Our company was founded in 2019 in San Francisco. Billing invoices are generated on the 1st of each month. Exceeding the rate limit returns HTTP 429 with a Retry-After header."
        ],
    )
    keep_ratio: float = Field(default=0.5, ge=0.1, le=1.0)
    min_score_threshold: float = Field(default=0.05, ge=0.0)


@app.post("/v1/rag/compress", summary="Compress Retrieved RAG Chunks")
def compress_rag_context(payload: CompressRequest):
    try:
        result = compressor.compress(
            query=payload.query,
            chunks=payload.chunks,
            keep_ratio=payload.keep_ratio,
            min_score_threshold=payload.min_score_threshold,
        )
        return {"status": "success", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compression error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
