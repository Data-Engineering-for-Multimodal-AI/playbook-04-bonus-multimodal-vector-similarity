#!/usr/bin/env python3
"""
Quality Check API for ByteEats Vector Similarity Search

This module provides REST API endpoints specifically for food quality assessment
by comparing prepared food images against reference images.
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add parent directory to path to import modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

from src.utils.quality_checker import QualityChecker

# Configuration
REFERENCE_IMAGE_DIR = os.environ.get("IMAGE_DIR", "data/images")
VIDEO_DIR = os.environ.get("VIDEO_DIR", "data/videos")
QUALITY_REPORT_DIR = os.environ.get("QUALITY_DIR", "data/quality")

# Ensure directories exist
os.makedirs(REFERENCE_IMAGE_DIR, exist_ok=True)
os.makedirs(QUALITY_REPORT_DIR, exist_ok=True)

# Pydantic models for request/response
class QualityCheckRequest(BaseModel):
    video_id: str
    reference_image: str
    threshold: float = 0.75

class QualityCheckResponse(BaseModel):
    video_id: str
    reference_image: str
    quality_passed: bool
    similarity_score: float = Field(alias="max_similarity_score")
    threshold: float
    timestamp: str
    assessment: str
    processing_time_seconds: float

# Initialize router
quality_router = APIRouter(
    prefix="/api/quality",
    tags=["quality"],
    responses={404: {"description": "Not found"}},
)

# Initialize quality checker
quality_checker = None

def init_quality_checker():
    """Initialize the quality checker."""
    global quality_checker
    if quality_checker is None:
        quality_checker = QualityChecker()
    return quality_checker

@quality_router.post("/check", response_model=QualityCheckResponse)
async def check_quality(request: QualityCheckRequest):
    """Check the quality of a video against a reference image."""
    init_quality_checker()
    
    # Find the video file
    video_file = None
    for f in os.listdir(VIDEO_DIR):
        if f.startswith(request.video_id) and f.endswith(('.mp4', '.avi', '.mov')):
            video_file = f
            break
    
    if not video_file:
        raise HTTPException(status_code=404, detail=f"Video ID {request.video_id} not found")
    
    # Find the reference image
    image_file = None
    for f in os.listdir(REFERENCE_IMAGE_DIR):
        if f == request.reference_image or (request.reference_image in f and f.endswith(('.jpg', '.jpeg', '.png'))):
            image_file = f
            break
    
    if not image_file:
        raise HTTPException(status_code=404, detail=f"Reference image {request.reference_image} not found")
    
    video_path = os.path.join(VIDEO_DIR, video_file)
    image_path = os.path.join(REFERENCE_IMAGE_DIR, image_file)
    
    try:
        # Assess quality
        start_time = time.time()
        report = quality_checker.assess_quality(video_path, image_path, request.threshold)
        
        if not report:
            raise HTTPException(status_code=500, detail="Failed to generate quality assessment")
        
        # Ensure all values are JSON serializable
        for key, value in report.items():
            if isinstance(value, (np.float32, np.float64)):
                report[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                report[key] = int(value)
            elif isinstance(value, np.ndarray):
                report[key] = value.tolist()
        
        # Ensure compatibility with the QualityCheckResponse model
        if 'max_similarity_score' in report and 'similarity_score' not in report:
            report['similarity_score'] = report['max_similarity_score']
        
        # Add processing time
        report["processing_time_seconds"] = round(time.time() - start_time, 3)
        
        # Save the report
        report_file = os.path.join(QUALITY_REPORT_DIR, f"{request.video_id}_quality.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        return report
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error assessing quality: {str(e)}")

@quality_router.post("/upload_reference")
async def upload_reference_image(
    image_file: UploadFile = File(...),
    image_name: str = Form(None)
):
    """Upload a reference image for quality comparison."""
    if not image_file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save the image
    os.makedirs(REFERENCE_IMAGE_DIR, exist_ok=True)
    
    # Use provided name or original filename
    filename = image_name or image_file.filename
    if not any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
        filename += '.jpg'  # Add extension if missing
    
    image_path = os.path.join(REFERENCE_IMAGE_DIR, filename)
    
    with open(image_path, "wb") as f:
        f.write(await image_file.read())
    
    return {
        "status": "success",
        "filename": filename,
        "message": "Reference image uploaded successfully"
    }

@quality_router.get("/reports")
async def get_quality_reports():
    """Get all quality assessment reports."""
    if not os.path.exists(QUALITY_REPORT_DIR):
        return {"reports": []}
    
    reports = []
    for f in os.listdir(QUALITY_REPORT_DIR):
        if f.endswith('_quality.json'):
            report_path = os.path.join(QUALITY_REPORT_DIR, f)
            try:
                with open(report_path, 'r') as file:
                    report = json.load(file)
                    reports.append(report)
            except Exception as e:
                print(f"Error loading report {f}: {str(e)}")
    
    return {"reports": reports}

@quality_router.get("/reference_images")
async def list_reference_images():
    """List all available reference images."""
    if not os.path.exists(REFERENCE_IMAGE_DIR):
        return {"images": [], "count": 0}
    
    images = [f for f in os.listdir(REFERENCE_IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    return {
        "images": images,
        "count": len(images)
    } 