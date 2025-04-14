#!/usr/bin/env python3
"""
Standalone server for ByteEats Quality Check API

This module runs the quality check API as a standalone server.
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path to import modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

# Import locally
from src.api.quality_check_api import quality_router

# API Configuration
PORT = int(os.environ.get("PORT", 8080))
IMAGE_DIR = os.environ.get("IMAGE_DIR", "data/images")
VIDEO_DIR = os.environ.get("VIDEO_DIR", "data/videos")
QUALITY_DIR = os.environ.get("QUALITY_DIR", "data/quality")

# Create FastAPI app
app = FastAPI(
    title="ByteEats Quality Check API",
    description="API for assessing food preparation quality using visual comparison",
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

# Include quality router
app.include_router(quality_router)

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "ByteEats Quality Check API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": [
            "/api/quality/check",
            "/api/quality/upload_reference",
            "/api/quality/reports",
            "/api/quality/reference_images"
        ],
        "environment": {
            "image_dir": IMAGE_DIR,
            "video_dir": VIDEO_DIR,
            "quality_dir": QUALITY_DIR
        }
    }

def main():
    """Run the API server."""
    print(f"Starting ByteEats Quality Check API on port {PORT}")
    print(f"Using:")
    print(f"  Image directory: {IMAGE_DIR}")
    print(f"  Video directory: {VIDEO_DIR}")
    print(f"  Quality reports directory: {QUALITY_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main() 