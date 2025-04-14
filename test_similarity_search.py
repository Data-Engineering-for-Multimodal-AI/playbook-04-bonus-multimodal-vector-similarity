#!/usr/bin/env python3
"""
Test script for ByteEats Vector Similarity Search Engine

This script tests the functionality of the Similarity Search Engine
by performing various similarity searches and analyzing the results.
"""

import os
import json
import numpy as np
from src.similarity.search import SimilaritySearch

# Configuration
EMBEDDINGS_DIR = "data/embeddings"
IMAGE_NAMES = ['Crispy_Shoyu', 'Tuna_Lover', 'Salmon_Edamame', 'Miso_Chicken', 
               'Kid\'s_Crab_Bowl', 'Miso_Salmon', 'House_Salmon']

def test_video_similarity():
    """Test video-to-video similarity search"""
    # Initialize search engine
    search_engine = SimilaritySearch(EMBEDDINGS_DIR)
    
    # Load embeddings
    num_embeddings = search_engine.load_embeddings()
    if num_embeddings == 0:
        print("No embeddings found. Make sure you've run the embedding generator.")
        return
    
    # Build index
    search_engine.build_index()
    
    # Choose a video ID to search for similar videos
    video_id = search_engine.video_ids[0]
    print(f"\nFinding videos similar to {video_id}:")
    
    # Find similar videos
    results, stats = search_engine.find_similar_by_id(video_id, k=5)
    
    # Print results
    for result in results:
        print(f"  {result['rank']}. {result['video_id']}: {result['similarity_score']:.4f}")
    
    print(f"\nSearch completed in {stats['search_time_ms']:.3f} ms")

def test_image_to_video_similarity():
    """Test image-to-video similarity search"""
    # Initialize search engine
    search_engine = SimilaritySearch(EMBEDDINGS_DIR)
    
    # Load embeddings
    search_engine.load_embeddings()
    
    # Build index
    search_engine.build_index()
    
    # Get a list of all image IDs
    image_ids = [id for id in search_engine.video_ids if any(img in id for img in IMAGE_NAMES)]
    
    if not image_ids:
        print("No image embeddings found.")
        return
    
    # Test similarity search for each image
    for image_id in image_ids:
        print(f"\nFinding videos similar to image {image_id}:")
        
        # Find similar videos
        results, stats = search_engine.find_similar_by_id(image_id, k=3)
        
        # Print results
        for result in results:
            print(f"  {result['rank']}. {result['video_id']}: {result['similarity_score']:.4f}")

def analyze_similarity_clustering():
    """Analyze embedding clusters to identify similar content groups"""
    # Initialize search engine
    search_engine = SimilaritySearch(EMBEDDINGS_DIR)
    
    # Load embeddings
    search_engine.load_embeddings()
    
    # Build index
    search_engine.build_index()
    
    # Analyze embeddings
    clusters = search_engine.analyze_embeddings()
    
    # Print cluster analysis
    print("\nCluster Analysis:")
    for cluster_id, cluster_info in enumerate(clusters.get("clusters", [])):
        members = cluster_info.get("members", [])
        print(f"\nCluster {cluster_id+1} with {len(members)} members:")
        for member in members[:5]:  # Show up to 5 members per cluster
            print(f"  - {member}")
        
        if len(members) > 5:
            print(f"  - ... and {len(members) - 5} more")

def test_custom_query():
    """Test a custom query using a specific embedding"""
    # Initialize search engine
    search_engine = SimilaritySearch(EMBEDDINGS_DIR)
    
    # Load embeddings
    search_engine.load_embeddings()
    
    # Build index
    search_engine.build_index()
    
    # Find a reference image to use as query
    image_ids = [id for id in search_engine.video_ids if any(img in id for img in IMAGE_NAMES)]
    
    if not image_ids:
        print("No image embeddings found for custom query.")
        return
    
    # Get the embedding for the reference image
    image_id = image_ids[0]
    image_idx = search_engine.video_ids.index(image_id)
    query_vector = search_engine.embeddings[image_idx]
    
    print(f"\nCustom query using {image_id} embedding:")
    
    # Add some noise to create a slightly different query
    noisy_query = query_vector + np.random.normal(0, 0.05, query_vector.shape)
    noisy_query = noisy_query / np.linalg.norm(noisy_query)  # Normalize
    
    # Find similar videos to the noisy query
    results, stats = search_engine.find_similar(noisy_query, k=5)
    
    # Print results
    for result in results:
        print(f"  {result['rank']}. {result['video_id']}: {result['similarity_score']:.4f}")
    
    print(f"\nSearch completed in {stats['search_time_ms']:.3f} ms")

def main():
    print("Testing ByteEats Vector Similarity Search Engine")
    print("=" * 50)
    
    # Test video-to-video similarity
    test_video_similarity()
    
    # Test image-to-video similarity
    test_image_to_video_similarity()
    
    # Test custom query
    test_custom_query()
    
    # Analyze clustering - this might be slow with many embeddings
    # Uncomment if you want to run the cluster analysis
    # analyze_similarity_clustering()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 