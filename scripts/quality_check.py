#!/usr/bin/env python3
"""
Quality Check Demo Script for ByteEats

This script demonstrates how to use the quality check API
to compare food preparation videos against reference images.
"""

import os
import sys
import json
import requests
import argparse
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Default API URL
API_URL = os.environ.get("API_URL", "http://localhost:8080")

def list_reference_images():
    """List all available reference images."""
    try:
        response = requests.get(f"{API_URL}/api/quality/reference_images")
        if response.status_code == 200:
            data = response.json()
            images = data.get("images", [])
            
            if not images:
                print("No reference images found.")
                return []
            
            print(f"Found {len(images)} reference images:")
            for i, image in enumerate(images, 1):
                print(f"  {i}. {image}")
            
            return images
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error listing reference images: {str(e)}")
        return []

def list_videos():
    """List all available videos."""
    try:
        response = requests.get(f"{API_URL}/api/videos")
        if response.status_code == 200:
            videos = response.json()
            
            if not videos:
                print("No videos found.")
                return []
            
            # Filter for only video type entries
            video_entries = [v.get('video_id') for v in videos if v.get('media_type') == 'video']
            
            if not video_entries:
                print("No videos found.")
                return []
                
            print(f"Found {len(video_entries)} videos:")
            for i, video in enumerate(video_entries, 1):
                print(f"  {i}. {video}")
            
            return video_entries
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error listing videos: {str(e)}")
        return []

def check_quality(video_id, reference_image, threshold=0.75):
    """Check the quality of a video against a reference image."""
    try:
        print(f"\nChecking quality of {video_id} against {reference_image}...")
        
        payload = {
            "video_id": video_id,
            "reference_image": reference_image,
            "threshold": threshold
        }
        
        response = requests.post(f"{API_URL}/api/quality/check", json=payload)
        
        if response.status_code == 200:
            report = response.json()
            
            # Print quality assessment results
            passed = report.get("quality_passed", False)
            score = report.get("max_similarity_score", 0)
            
            print("\n" + "="*50)
            print(f"QUALITY CHECK: {'PASSED' if passed else 'FAILED'}")
            print("="*50)
            print(f"Video: {video_id}")
            print(f"Reference Image: {reference_image}")
            print(f"Similarity Score: {score:.4f} (threshold: {threshold:.2f})")
            print(f"Assessment: {report.get('assessment', 'Unknown')}")
            print(f"Processing Time: {report.get('processing_time_seconds', 0):.3f} seconds")
            print("="*50)
            
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error checking quality: {str(e)}")
        return False

def get_quality_reports():
    """Get all quality assessment reports."""
    try:
        response = requests.get(f"{API_URL}/api/quality/reports")
        if response.status_code == 200:
            data = response.json()
            reports = data.get("reports", [])
            
            if not reports:
                print("No quality reports found.")
                return []
            
            print(f"\nFound {len(reports)} quality reports:")
            
            # Create a summary table
            print("\n{:<15} {:<25} {:<15} {:<10} {:<10}".format(
                "VIDEO", "REFERENCE", "SCORE", "THRESHOLD", "RESULT"
            ))
            print("-"*75)
            
            for report in reports:
                video_id = report.get("video_id", "")
                ref_img = report.get("reference_image", "")
                score = report.get("max_similarity_score", 0)
                threshold = report.get("threshold", 0)
                passed = report.get("quality_passed", False)
                
                result = "PASSED" if passed else "FAILED"
                
                print("{:<15} {:<25} {:<15.4f} {:<10.2f} {:<10}".format(
                    video_id, ref_img, score, threshold, result
                ))
            
            return reports
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error getting quality reports: {str(e)}")
        return []

def interactive_mode():
    """Run in interactive mode to check quality of videos against reference images."""
    print("\n=== ByteEats Quality Check Tool ===\n")
    
    # List videos
    videos = list_videos()
    if not videos:
        print("No videos available for quality checking.")
        return
    
    # List reference images
    images = list_reference_images()
    if not images:
        print("No reference images available for quality checking.")
        return
    
    # Get video selection
    print("\nSelect a video to check (enter number):")
    try:
        video_idx = int(input("> ")) - 1
        if video_idx < 0 or video_idx >= len(videos):
            print("Invalid selection.")
            return
        selected_video = videos[video_idx]
    except ValueError:
        print("Please enter a valid number.")
        return
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return
    
    # Get reference image selection
    print("\nSelect a reference image (enter number):")
    try:
        image_idx = int(input("> ")) - 1
        if image_idx < 0 or image_idx >= len(images):
            print("Invalid selection.")
            return
        selected_image = images[image_idx]
    except ValueError:
        print("Please enter a valid number.")
        return
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return
    
    # Get threshold
    print("\nEnter quality threshold (0.5-0.95, default 0.75):")
    try:
        threshold_input = input("> ")
        threshold = 0.75 if not threshold_input else float(threshold_input)
        if threshold < 0.5 or threshold > 0.95:
            print("Threshold must be between 0.5 and 0.95. Using default 0.75.")
            threshold = 0.75
    except ValueError:
        print("Invalid threshold. Using default 0.75.")
        threshold = 0.75
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return
    
    # Perform quality check
    check_quality(selected_video, selected_image, threshold)
    
    # Show all reports
    print("\nWould you like to see all quality reports? (y/n)")
    try:
        choice = input("> ").lower()
        if choice.startswith('y'):
            get_quality_reports()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ByteEats Quality Check Tool")
    parser.add_argument("--list-videos", action="store_true", help="List available videos")
    parser.add_argument("--list-images", action="store_true", help="List available reference images")
    parser.add_argument("--video", help="Video ID to check")
    parser.add_argument("--reference", help="Reference image to compare against")
    parser.add_argument("--threshold", type=float, default=0.75, help="Quality threshold (0.5-0.95)")
    parser.add_argument("--reports", action="store_true", help="Show all quality reports")
    parser.add_argument("--api-url", help="API URL (default: http://localhost:8080)")
    
    args = parser.parse_args()
    
    # Set API URL if provided
    global API_URL
    if args.api_url:
        API_URL = args.api_url
    
    print(f"Using API URL: {API_URL}")
    
    # Process command line arguments
    if args.list_videos:
        list_videos()
    elif args.list_images:
        list_reference_images()
    elif args.reports:
        get_quality_reports()
    elif args.video and args.reference:
        check_quality(args.video, args.reference, args.threshold)
    else:
        interactive_mode()

if __name__ == "__main__":
    main() 