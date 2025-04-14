#!/usr/bin/env python3
"""
Directory Setup Script for ByteEats

This script ensures all necessary directories exist and are properly
structured for the ByteEats Vector Similarity Search system.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Directory structure
DIRS = {
    "data": [
        "videos",
        "embeddings",
        "images",
        "quality"
    ],
    "logs": [],
    "models": [],
    "src": [
        "api",
        "embeddings",
        "similarity",
        "utils"
    ],
    "scripts": [],
    "tests": []
}

def create_directory_structure():
    """Create the necessary directory structure."""
    print("Creating directory structure...")
    
    for parent, children in DIRS.items():
        parent_path = os.path.join(parent_dir, parent)
        os.makedirs(parent_path, exist_ok=True)
        print(f"✓ Created {parent}/")
        
        for child in children:
            child_path = os.path.join(parent_path, child)
            os.makedirs(child_path, exist_ok=True)
            print(f"  ✓ Created {parent}/{child}/")
    
    print("Directory structure created successfully.")

def list_reference_images():
    """List reference images."""
    reference_dir = os.path.join(parent_dir, "data", "images")
    
    if not os.path.exists(reference_dir):
        print(f"Reference images directory not found: {reference_dir}")
        return []
    
    images = [f for f in os.listdir(reference_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print("No reference images found.")
    else:
        print(f"Found {len(images)} reference images:")
        for i, image in enumerate(images, 1):
            print(f"  {i}. {image}")
    
    return images

def check_image_paths():
    """Check if image paths match environment variables."""
    # Check environment variables
    image_dir_env = os.environ.get("IMAGE_DIR", "")
    print(f"IMAGE_DIR environment variable: {image_dir_env or 'Not set'}")
    
    default_image_dir = os.path.join(parent_dir, "data", "images")
    
    print(f"Default reference images directory: {default_image_dir}")
    print(f"Using directory: {image_dir_env or default_image_dir}")
    
    # Check directory existence
    if image_dir_env and not os.path.exists(image_dir_env):
        print(f"Warning: IMAGE_DIR directory does not exist: {image_dir_env}")
        print(f"Creating directory: {image_dir_env}")
        os.makedirs(image_dir_env, exist_ok=True)
    
    # Count images
    dir_to_check = image_dir_env or default_image_dir
    if os.path.exists(dir_to_check):
        images = [f for f in os.listdir(dir_to_check) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Number of reference images: {len(images)}")
    else:
        print(f"Warning: Reference images directory does not exist: {dir_to_check}")

def move_images(source_dir):
    """Move images from source directory to reference_images directory."""
    if not source_dir or not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return False
    
    target_dir = os.path.join(parent_dir, "data", "images")
    os.makedirs(target_dir, exist_ok=True)
    
    # Get images from source directory
    images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print(f"No images found in source directory: {source_dir}")
        return False
    
    print(f"Found {len(images)} images in {source_dir}")
    print(f"Moving images to {target_dir}...")
    
    num_moved = 0
    for image in images:
        source_path = os.path.join(source_dir, image)
        target_path = os.path.join(target_dir, image)
        
        try:
            shutil.copy2(source_path, target_path)
            print(f"  ✓ Copied {image}")
            num_moved += 1
        except Exception as e:
            print(f"  ✗ Error copying {image}: {str(e)}")
    
    print(f"Moved {num_moved} out of {len(images)} images.")
    return num_moved > 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ByteEats Directory Setup Tool")
    parser.add_argument("--create", action="store_true", help="Create directory structure")
    parser.add_argument("--list-images", action="store_true", help="List reference images")
    parser.add_argument("--check-paths", action="store_true", help="Check image paths")
    parser.add_argument("--move-images", help="Move images from specified directory to reference_images")
    
    args = parser.parse_args()
    
    if args.create:
        create_directory_structure()
    elif args.list_images:
        list_reference_images()
    elif args.check_paths:
        check_image_paths()
    elif args.move_images:
        move_images(args.move_images)
    else:
        # Default: create directory structure and check paths
        create_directory_structure()
        check_image_paths()
        list_reference_images()

if __name__ == "__main__":
    main() 