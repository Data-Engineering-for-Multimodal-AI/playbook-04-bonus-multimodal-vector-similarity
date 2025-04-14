#!/usr/bin/env python3
"""
FastAPI Application for ByteEats Vector Similarity Search

This module provides a REST API for video similarity search.
"""

import os
import sys
import time
import json
import uvicorn
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime

# Add parent directory to path to import modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

from src.embeddings.generator import VideoEmbeddingGenerator
from src.similarity.search import SimilaritySearch
from src.utils.quality_checker import QualityChecker
from src.api.quality_check_api import quality_router

# API models
class SimilarityRequest(BaseModel):
    video_id: str
    k: int = 5

class VectorRequest(BaseModel):
    embedding: List[float]
    k: int = 5

class SearchResponse(BaseModel):
    query: Dict[str, Any]
    results: List[Dict[str, Any]]
    stats: Dict[str, Any]

# Initialize the API
app = FastAPI(
    title="ByteEats Vector Similarity Search API",
    description="API for searching similar food preparation videos using vector embeddings",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include quality check router
app.include_router(quality_router)

# Global variables
embedding_dir = os.environ.get("EMBEDDING_DIR", "data/embeddings")
video_dir = os.environ.get("VIDEO_DIR", "data/videos")
image_dir = os.environ.get("IMAGE_DIR", "data/images")
quality_dir = os.environ.get("QUALITY_DIR", "data/quality")
similarity_search = None
embedding_generator = None
quality_checker = None

def initialize_api():
    """Initialize the API components."""
    global similarity_search, embedding_generator, quality_checker
    
    # Check if embeddings directory exists
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir, exist_ok=True)
        print(f"Created embeddings directory: {embedding_dir}")
    
    # Check if quality directory exists
    if not os.path.exists(quality_dir):
        os.makedirs(quality_dir, exist_ok=True)
        print(f"Created quality directory: {quality_dir}")
    
    # Initialize similarity search
    similarity_search = SimilaritySearch(embedding_dir)
    num_embeddings = similarity_search.load_embeddings()
    
    if num_embeddings > 0:
        similarity_search.build_index(index_type="Flat")
    
    # Initialize embedding generator
    embedding_generator = VideoEmbeddingGenerator()
    
    # Initialize quality checker
    quality_checker = QualityChecker()
    
    print(f"API initialized with {num_embeddings} embeddings")
    return num_embeddings

@app.on_event("startup")
async def startup_event():
    """Initialize the similarity search and quality checker on startup."""
    initialize_api()

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "ByteEats Vector Similarity Search API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": [
            "/api/videos",
            "/api/similar/{video_id}",
            "/api/search",
            "/api/embedding/{video_id}",
            "/api/stats",
            "/api/quality/check",
            "/api/quality/upload_reference",
            "/api/quality/reports"
        ]
    }

@app.get("/api/videos")
async def list_videos():
    """List all available videos."""
    global similarity_search
    
    if not similarity_search or not similarity_search.video_ids:
        return []
    
    # Format video information
    video_list = []
    for i, video_id in enumerate(similarity_search.video_ids):
        video_info = {
            "video_id": video_id,
            "media_type": "unknown"
        }
        
        # Add metadata if available
        if i < len(similarity_search.metadata) and similarity_search.metadata[i]:
            metadata = similarity_search.metadata[i]
            video_info.update({
                "media_type": metadata.get("media_type", "unknown"),
                "timestamp": metadata.get("timestamp", ""),
                "file_size_mb": metadata.get("file_size_mb", 0),
                "processing_time_seconds": metadata.get("processing_time_seconds", 0)
            })
        
        video_list.append(video_info)
    
    return video_list

@app.get("/api/similar/{video_id}")
async def find_similar_videos(video_id: str, k: int = 5):
    """Find similar videos to the specified video."""
    global similarity_search
    
    if not similarity_search:
        raise HTTPException(status_code=503, detail="Similarity search not initialized")
    
    if video_id not in similarity_search.video_ids:
        raise HTTPException(status_code=404, detail=f"Video ID {video_id} not found")
    
    results, stats = similarity_search.find_similar_by_id(video_id, k=k)
    
    return {
        "query": {"video_id": video_id, "k": k},
        "results": results,
        "stats": stats
    }

@app.post("/api/search")
async def search_by_vector(request: VectorRequest):
    """Find similar videos based on a vector."""
    global similarity_search
    
    if not similarity_search:
        raise HTTPException(status_code=503, detail="Similarity search not initialized")
    
    try:
        query_vector = np.array(request.embedding).astype('float32')
        results, stats = similarity_search.find_similar(query_vector, k=request.k)
        
        return {
            "query": {"embedding_dimension": len(request.embedding), "k": request.k},
            "results": results,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid vector format: {str(e)}")

@app.get("/api/embedding/{video_id}")
async def get_embedding(video_id: str):
    """Get the embedding for a specific video."""
    global similarity_search
    
    if not similarity_search:
        raise HTTPException(status_code=503, detail="Similarity search not initialized")
    
    if video_id not in similarity_search.video_ids:
        raise HTTPException(status_code=404, detail=f"Video ID {video_id} not found")
    
    idx = similarity_search.video_ids.index(video_id)
    embedding = similarity_search.embeddings[idx].tolist()
    
    return {
        "video_id": video_id,
        "embedding_dimension": len(embedding),
        "embedding": embedding
    }

@app.get("/api/stats")
async def get_stats():
    """Get statistics about the API and embeddings."""
    global similarity_search
    
    if not similarity_search:
        return {
            "status": "not_initialized",
            "timestamp": datetime.now().isoformat()
        }
    
    # Count image and video embeddings
    video_count = 0
    image_count = 0
    
    for i, metadata in enumerate(similarity_search.metadata):
        media_type = metadata.get("media_type", "unknown")
        if media_type == "video":
            video_count += 1
        elif media_type == "image":
            image_count += 1
    
    return {
        "status": "initialized",
        "timestamp": datetime.now().isoformat(),
        "embedding_count": len(similarity_search.embeddings),
        "video_count": video_count,
        "image_count": image_count,
        "index_type": "Flat",
        "embedding_dimension": similarity_search.embeddings[0].shape[0] if len(similarity_search.embeddings) > 0 else 0
    }

def main():
    """Run the API server."""
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting ByteEats Vector Similarity Search API on port {port}")
    initialize_api()
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main() 