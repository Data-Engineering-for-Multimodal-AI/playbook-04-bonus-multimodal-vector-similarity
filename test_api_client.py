#!/usr/bin/env python3
"""
Test client for ByteEats Vector Similarity Search API

This script tests the API endpoints for similarity search.
"""

import requests
import json
import sys
import os
from pprint import pprint

# API Configuration
API_PORT = os.environ.get("API_PORT", "8085")
API_URL = f"http://localhost:{API_PORT}"

def test_list_videos():
    """Test the endpoint to list available videos"""
    endpoint = f"{API_URL}/api/videos"
    print(f"Testing GET {endpoint}")
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data)} videos")
            if len(data) > 0:
                print("Sample video data:")
                pprint(data[0])
            return data
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Request error: {str(e)}")
        return None

def test_similar_videos(video_id):
    """Test finding similar videos by ID"""
    endpoint = f"{API_URL}/api/similar/{video_id}"
    print(f"\nTesting GET {endpoint}")
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data['results'])} similar videos")
            print("Similar videos:")
            for result in data['results']:
                print(f"  {result['rank']}. {result['video_id']}: {result['similarity_score']:.4f}")
            return data
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Request error: {str(e)}")
        return None

def test_search_by_embedding(video_id):
    """Test searching by getting an embedding and using it for search"""
    # First get the embedding for a video
    embedding_endpoint = f"{API_URL}/api/embedding/{video_id}"
    print(f"\nTesting GET {embedding_endpoint}")
    
    try:
        response = requests.get(embedding_endpoint)
        if response.status_code != 200:
            print(f"Error getting embedding: {response.status_code}")
            print(response.text)
            return None
        
        embedding_data = response.json()
        if "embedding" not in embedding_data:
            print("No embedding found in response")
            return None
        
        embedding = embedding_data["embedding"]
        print(f"Got embedding with {len(embedding)} dimensions")
        
        # Now use this embedding to search
        search_endpoint = f"{API_URL}/api/search"
        print(f"\nTesting POST {search_endpoint}")
        
        search_payload = {
            "embedding": embedding,
            "k": 5  # Number of results to return
        }
        
        search_response = requests.post(search_endpoint, json=search_payload)
        if search_response.status_code == 200:
            search_data = search_response.json()
            print(f"Found {len(search_data['results'])} similar videos")
            print("Similar videos:")
            for result in search_data['results']:
                print(f"  {result['rank']}. {result['video_id']}: {result['similarity_score']:.4f}")
            return search_data
        else:
            print(f"Error in search: {search_response.status_code}")
            print(search_response.text)
            return None
        
    except Exception as e:
        print(f"Request error: {str(e)}")
        return None

def test_stats():
    """Test getting API stats"""
    endpoint = f"{API_URL}/api/stats"
    print(f"\nTesting GET {endpoint}")
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            print("API Stats:")
            pprint(data)
            return data
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Request error: {str(e)}")
        return None

def main():
    # Make sure API_PORT is set
    if "API_PORT" not in os.environ:
        print("Warning: API_PORT environment variable not set, using default 8085")
    
    print(f"Testing ByteEats Vector Similarity Search API at {API_URL}")
    print("=" * 60)
    
    # Get list of videos
    videos = test_list_videos()
    
    if not videos:
        print("No videos found or API not available. Exiting.")
        return
    
    # Get a sample video ID
    video_id = videos[0]["video_id"] if videos and len(videos) > 0 else None
    
    if not video_id:
        print("No video ID found for testing. Exiting.")
        return
    
    # Test similar videos endpoint
    test_similar_videos(video_id)
    
    # Test search by embedding
    test_search_by_embedding(video_id)
    
    # Test stats endpoint
    test_stats()
    
    print("\nAll API tests completed!")

if __name__ == "__main__":
    main() 