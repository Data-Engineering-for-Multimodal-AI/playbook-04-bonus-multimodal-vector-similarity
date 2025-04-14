#!/usr/bin/env python3
"""
Analysis script for ByteEats Vector Similarity Search embeddings
"""

import os
import numpy as np
import json
from collections import defaultdict

# Configuration
EMBEDDINGS_DIR = "data/embeddings"
IMAGE_NAMES = ['Crispy_Shoyu', 'Tuna_Lover', 'Salmon_Edamame', 'Miso_Chicken', 
               'Kid\'s_Crab_Bowl', 'Miso_Salmon', 'House_Salmon']

def load_embeddings():
    """Load all embeddings and separate them by type (video/image)"""
    video_embeddings = {}
    image_embeddings = {}
    
    for filename in os.listdir(EMBEDDINGS_DIR):
        if filename.endswith('_embedding.npy'):
            media_id = filename.replace('_embedding.npy', '')
            embedding_path = os.path.join(EMBEDDINGS_DIR, filename)
            metadata_path = os.path.join(EMBEDDINGS_DIR, f"{media_id}_metadata.json")
            
            # Load embedding
            embedding = np.load(embedding_path)
            
            # Check if this is an image or video by name matching
            is_image = any(img_name in media_id for img_name in IMAGE_NAMES)
            
            # Load metadata if available
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            if is_image:
                image_embeddings[media_id] = {
                    'embedding': embedding,
                    'metadata': metadata
                }
            else:
                video_embeddings[media_id] = {
                    'embedding': embedding,
                    'metadata': metadata
                }
    
    return video_embeddings, image_embeddings

def compute_similarities(video_embeddings, image_embeddings):
    """Compute similarities between all videos and images"""
    similarities = defaultdict(dict)
    
    for video_id, video_data in video_embeddings.items():
        v_emb = video_data['embedding']
        for image_id, image_data in image_embeddings.items():
            i_emb = image_data['embedding']
            
            # Calculate cosine similarity
            similarity = np.dot(v_emb, i_emb) / (np.linalg.norm(v_emb) * np.linalg.norm(i_emb))
            similarities[video_id][image_id] = similarity
    
    return similarities

def analyze_embeddings():
    """Analyze the embeddings directory and print statistics"""
    # Load embeddings
    video_embeddings, image_embeddings = load_embeddings()
    
    print(f"Found {len(video_embeddings)} video embeddings and {len(image_embeddings)} image embeddings")
    
    # Check embedding dimensions
    if video_embeddings and image_embeddings:
        v_id = next(iter(video_embeddings))
        i_id = next(iter(image_embeddings))
        v_emb = video_embeddings[v_id]['embedding']
        i_emb = image_embeddings[i_id]['embedding']
        
        print(f"Video embedding shape: {v_emb.shape}")
        print(f"Image embedding shape: {i_emb.shape}")
        
        # Verify they have the same shape for compatibility
        if v_emb.shape == i_emb.shape:
            print(f"✓ Embeddings have compatible shapes")
        else:
            print(f"✗ Warning: Embeddings have incompatible shapes")
    
    # Compute similarities
    similarities = compute_similarities(video_embeddings, image_embeddings)
    
    # Print top similarities for each image
    print("\nTop video matches for each reference image:")
    for image_id in image_embeddings:
        top_videos = []
        for video_id, similarity_dict in similarities.items():
            if image_id in similarity_dict:
                top_videos.append((video_id, similarity_dict[image_id]))
        
        # Sort by similarity (descending)
        top_videos.sort(key=lambda x: x[1], reverse=True)
        
        # Print top 3 most similar videos for this image
        print(f"\n{image_id}:")
        for i, (video_id, similarity) in enumerate(top_videos[:3]):
            print(f"  {i+1}. {video_id}: {similarity:.4f}")

    # Calculate average embedding values
    video_avg = np.mean([data['embedding'] for data in video_embeddings.values()], axis=0)
    image_avg = np.mean([data['embedding'] for data in image_embeddings.values()], axis=0)
    
    print("\nEmbedding statistics:")
    print(f"Average video embedding magnitude: {np.linalg.norm(video_avg):.4f}")
    print(f"Average image embedding magnitude: {np.linalg.norm(image_avg):.4f}")
    
    # Similarity between average video and average image embedding
    avg_similarity = np.dot(video_avg, image_avg) / (np.linalg.norm(video_avg) * np.linalg.norm(image_avg))
    print(f"Similarity between average video and image embedding: {avg_similarity:.4f}")

if __name__ == "__main__":
    analyze_embeddings() 