#!/usr/bin/env python3
"""
Test client for ByteEats Quality Check API

This script tests the quality check API endpoints.
"""

import os
import json
import requests
from pprint import pprint

# API Configuration
API_PORT = os.environ.get("API_PORT", "8085")
API_URL = f"http://localhost:{API_PORT}"

def test_server_info():
    """Test the root endpoint to verify server is running"""
    endpoint = f"{API_URL}/"
    print(f"Testing GET {endpoint}")
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            print("Server is running. API Info:")
            pprint(response.json())
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Request error: {str(e)}")
        return False

def test_list_reference_images():
    """Test listing reference images"""
    endpoint = f"{API_URL}/api/quality/reference_images"
    print(f"\nTesting GET {endpoint}")
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            print(f"Found {data['count']} reference images:")
            for i, image in enumerate(data['images']):
                print(f"  {i+1}. {image}")
            return data['images']
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"Request error: {str(e)}")
        return []

def test_quality_check(video_id, reference_image):
    """Test quality check endpoint"""
    endpoint = f"{API_URL}/api/quality/check"
    print(f"\nTesting POST {endpoint}")
    
    payload = {
        "video_id": video_id,
        "reference_image": reference_image,
        "threshold": 0.7
    }
    
    try:
        response = requests.post(endpoint, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"Quality check completed for {video_id} against {reference_image}:")
            print(f"  Passed: {'Yes' if result['quality_passed'] else 'No'}")
            print(f"  Similarity score: {result['max_similarity_score']:.4f}")
            print(f"  Assessment: {result['assessment']}")
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Request error: {str(e)}")
        return None

def test_quality_reports():
    """Test getting quality reports"""
    endpoint = f"{API_URL}/api/quality/reports"
    print(f"\nTesting GET {endpoint}")
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            reports = data.get("reports", [])
            print(f"Found {len(reports)} quality reports")
            if reports:
                print("Latest report:")
                pprint(reports[0])
            return reports
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"Request error: {str(e)}")
        return []

def main():
    print(f"Testing ByteEats Quality Check API at {API_URL}")
    print("=" * 60)
    
    # Test server info
    if not test_server_info():
        print("Server is not running or not accessible. Exiting.")
        return
    
    # Get available reference images
    reference_images = test_list_reference_images()
    if not reference_images:
        print("No reference images found. Please make sure images are available in the image directory.")
        return
    
    # Test quality check with first reference image
    reference_image = reference_images[0]
    
    # First, find an available video
    video_id = "download"  # Default test video
    
    # Run a quality check
    test_quality_check(video_id, reference_image)
    
    # Check for reports
    test_quality_reports()
    
    print("\nAll API tests completed!")

if __name__ == "__main__":
    main() 