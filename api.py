import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from typing import List, Optional
from functools import lru_cache
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cross-Encoder Reranking API",
    description=f"API for reranking documents using {os.getenv('MODEL_NAME', 'cross-encoder/ms-marco-MiniLM-L6-v2')} model",
    version="1.0.0"
)

# Request and Response Models
class QueryDocPair(BaseModel):
    query: str = Field(..., description="Search query")
    text: str = Field(..., description="Document text to be ranked")

class RerankRequest(BaseModel):
    query: str = Field(..., description="Search query")
    documents: List[str] = Field(..., description="List of documents to rerank", min_items=1)
    top_k: Optional[int] = Field(None, description="Return only top K results", ge=1)

class RerankResponse(BaseModel):
    query: str
    results: List[dict]

class ScoreRequest(BaseModel):
    pairs: List[QueryDocPair] = Field(..., description="List of query-document pairs", min_items=1)

class ScoreResponse(BaseModel):
    scores: List[float]

# Model loading with caching
@lru_cache(maxsize=1)
def load_model():
    """Load and cache the cross-encoder model"""
    try:
        logger.info("Loading cross-encoder model...")
        model = CrossEncoder(
            os.getenv("MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L12-v2"),
            max_length=512,
            device='cpu'
        )
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

# Startup event
@app.on_event("startup")
async def startup_event():
    """Preload model on startup"""
    load_model()
    logger.info("API is ready")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model = load_model()
        return {"status": "healthy", "model": os.getenv("MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L12-v2")}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Rerank endpoint
@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank documents based on query relevance.
    Returns documents sorted by relevance score in descending order.
    """
    try:
        model = load_model()
        
        # Create query-document pairs
        pairs = [[request.query, doc] for doc in request.documents]
        
        # Get scores with batch processing
        scores = model.predict(pairs, batch_size=int(os.getenv("BATCH_SIZE", "32")), show_progress_bar=False)
        
        # Combine documents with scores
        results = [
            {
                "index": idx,
                "text": doc,
                "score": float(score)
            }
            for idx, (doc, score) in enumerate(zip(request.documents, scores))
        ]
        
        # Sort by score in descending order
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply top_k filtering if specified
        if request.top_k:
            results = results[:request.top_k]
        
        return RerankResponse(query=request.query, results=results)
    
    except Exception as e:
        logger.error(f"Error during reranking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")

# Score endpoint for custom pairs
@app.post("/score", response_model=ScoreResponse)
async def score_pairs(request: ScoreRequest):
    """
    Score custom query-document pairs.
    Returns similarity scores for each pair.
    """
    try:
        model = load_model()
        
        # Create pairs from request
        pairs = [[pair.query, pair.text] for pair in request.pairs]
        
        # Get scores
        scores = model.predict(pairs, batch_size=int(os.getenv("BATCH_SIZE", "32")), show_progress_bar=False)
        
        return ScoreResponse(scores=[float(score) for score in scores])
    
    except Exception as e:
        logger.error(f"Error during scoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

# Batch rerank endpoint for multiple queries
@app.post("/batch_rerank")
async def batch_rerank(queries: List[RerankRequest]):
    """
    Rerank documents for multiple queries in a single request.
    Useful for processing multiple search results simultaneously.
    """
    try:
        results = []
        for query_request in queries:
            rerank_result = await rerank(query_request)
            results.append(rerank_result)
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Error during batch reranking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch reranking failed: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cross-Encoder Reranking API",
        "model": os.getenv("MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L12-v2"),
        "endpoints": {
            "/rerank": "Rerank documents for a query",
            "/score": "Score custom query-document pairs",
            "/batch_rerank": "Rerank for multiple queries",
            "/health": "Health check",
            "/docs": "API documentation"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=os.getenv("RERANKER_PORT", 8000),
        workers=1,
        log_level="info"
    )
