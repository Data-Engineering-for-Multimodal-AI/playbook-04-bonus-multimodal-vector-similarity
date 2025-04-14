#!/usr/bin/env python3
"""
Initialization Utilities for ByteEats Vector Similarity Search

This module provides utilities for setting up the project structure
and running the end-to-end pipeline.
"""

import os
import argparse
import shutil
from datetime import datetime

def setup_directory_structure(base_dir="."):
    """Create the standard directory structure for the project."""
    directories = [
        "data/videos",
        "data/images",
        "data/orders",
        "data/embeddings",
        "data/results",
        "models",
        "notebooks",
        "docs"
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return True

def copy_sample_data(source_dir, target_dir, file_types=None):
    """Copy sample data files from source to target directory."""
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return False
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Default file types if none specified
    if file_types is None:
        file_types = ['.mp4', '.avi', '.mov', '.jpg', '.png', '.json']
    
    # Count files copied
    count = 0
    
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        
        # Skip directories
        if os.path.isdir(source_file):
            continue
        
        # Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in file_types:
            continue
        
        # Copy the file
        target_file = os.path.join(target_dir, filename)
        shutil.copy2(source_file, target_file)
        print(f"Copied: {filename} to {target_dir}")
        count += 1
    
    print(f"Copied {count} files to {target_dir}")
    return count > 0

def generate_sample_queries(embeddings_dir, output_file):
    """Generate a sample queries file for batch search."""
    import numpy as np
    import json
    
    # Find all embedding files
    embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('_embedding.npy')]
    
    if not embedding_files:
        print(f"No embedding files found in {embeddings_dir}")
        return False
    
    # Generate sample queries
    queries = []
    
    # Add video ID based queries
    for i, embedding_file in enumerate(embedding_files[:5]):
        video_id = embedding_file.replace('_embedding.npy', '')
        queries.append({
            "query_id": f"query_{i+1}",
            "query_type": "video_id",
            "video_id": video_id,
            "description": f"Find videos similar to {video_id}"
        })
    
    # Add vector based queries if possible
    if len(embedding_files) > 0:
        embedding_path = os.path.join(embeddings_dir, embedding_files[0])
        try:
            embedding = np.load(embedding_path)
            
            # Add a slightly perturbed version of the embedding
            perturbed = embedding + 0.1 * np.random.randn(*embedding.shape)
            perturbed = perturbed / np.linalg.norm(perturbed)
            
            queries.append({
                "query_id": "vector_query_1",
                "query_type": "vector",
                "vector": perturbed.tolist(),
                "description": "Find videos similar to a perturbed embedding"
            })
            
        except Exception as e:
            print(f"Error loading embedding: {str(e)}")
    
    # Save the queries
    with open(output_file, 'w') as f:
        json.dump(queries, f, indent=4)
    
    print(f"Generated {len(queries)} sample queries saved to {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Initialize ByteEats Vector Similarity Search project")
    parser.add_argument("--setup", action="store_true", help="Setup directory structure")
    parser.add_argument("--copy-videos", help="Copy videos from source directory to data/videos")
    parser.add_argument("--copy-images", help="Copy images from source directory to data/images")
    parser.add_argument("--copy-orders", help="Copy order data from source directory to data/orders")
    parser.add_argument("--generate-queries", action="store_true", 
                        help="Generate sample queries for batch search")
    
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    if args.setup:
        setup_directory_structure(project_root)
    
    if args.copy_videos:
        target_dir = os.path.join(project_root, "data/videos")
        copy_sample_data(args.copy_videos, target_dir, ['.mp4', '.avi', '.mov'])
    
    if args.copy_images:
        target_dir = os.path.join(project_root, "data/images")
        copy_sample_data(args.copy_images, target_dir, ['.jpg', '.jpeg', '.png'])
    
    if args.copy_orders:
        target_dir = os.path.join(project_root, "data/orders")
        copy_sample_data(args.copy_orders, target_dir, ['.json', '.jsonl'])
    
    if args.generate_queries:
        embeddings_dir = os.path.join(project_root, "data/embeddings")
        output_file = os.path.join(project_root, "data/sample_queries.json")
        generate_sample_queries(embeddings_dir, output_file)
    
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main() 