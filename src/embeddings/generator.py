#!/usr/bin/env python3
"""
Media Embedding Generator for ByteEats Vector Similarity Search

This module extracts features from food preparation videos and images, then generates 
vector embeddings for similarity search.
"""

import os
import cv2
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import torchvision.transforms as transforms
from datetime import datetime
from PIL import Image

# Constants
FRAME_SAMPLE_RATE = 5  # Extract every 5th frame
EMBEDDING_SIZE = 512
OUTPUT_VERSION = "v1"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov')

class MediaEmbeddingGenerator:
    def __init__(self, model_name="resnet18", device=None):
        """Initialize the embedding generator with a pre-trained model."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load pre-trained model using the weights API (pretrained= was removed in torchvision 0.15+)
        if model_name == "resnet18":
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        elif model_name == "resnet50":
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transformations for PIL images
        self.pil_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Define image transformations for numpy arrays (frames)
        self.frame_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def extract_frames(self, video_path):
        """Extract frames from a video file."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        
        # Calculate frames to sample
        frames_to_sample = range(0, frame_count, FRAME_SAMPLE_RATE)
        
        for i in frames_to_sample:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames, duration
    
    def load_image(self, image_path):
        """Load an image from file."""
        try:
            # Load with PIL for better compatibility
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def generate_image_embedding(self, image_path):
        """Generate an embedding for an image file."""
        start_time = time.time()
        
        # Load image
        image = self.load_image(image_path)
        
        if image is None:
            print(f"Warning: Failed to load image {image_path}")
            return None, None
        
        # Process image
        with torch.no_grad():
            input_tensor = self.pil_transform(image).unsqueeze(0).to(self.device)
            feature = self.model(input_tensor)
            embedding = feature.squeeze().cpu().numpy()
        
        # Normalize the embedding vector
        embedding = embedding / np.linalg.norm(embedding)
        
        processing_time = time.time() - start_time
        
        return embedding, processing_time
    
    def generate_video_embedding(self, video_path):
        """Generate an embedding for a video file."""
        start_time = time.time()
        
        # Extract frames
        frames, duration = self.extract_frames(video_path)
        
        if not frames:
            print(f"Warning: No frames extracted from {video_path}")
            return None, None, 0
        
        # Process frames
        frame_embeddings = []
        
        with torch.no_grad():
            for frame in frames:
                input_tensor = self.frame_transform(frame).unsqueeze(0).to(self.device)
                feature = self.model(input_tensor)
                embedding = feature.squeeze().cpu().numpy()
                frame_embeddings.append(embedding)
        
        # Average the frame embeddings
        video_embedding = np.mean(frame_embeddings, axis=0)
        
        # Normalize the embedding vector
        video_embedding = video_embedding / np.linalg.norm(video_embedding)
        
        processing_time = time.time() - start_time
        
        return video_embedding, frames, processing_time
    
    def process_directory(self, input_dir, output_dir, limit=None):
        """Process a single directory containing media files."""
        results = []
        
        # Get all media files in this directory
        all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        video_files = [f for f in all_files if f.endswith(VIDEO_EXTENSIONS)]
        image_files = [f for f in all_files if f.endswith(IMAGE_EXTENSIONS)]
        
        print(f"In directory {input_dir}:")
        print(f"  Found {len(video_files)} video files and {len(image_files)} image files")
        
        # Apply limit if specified
        if limit:
            video_files = video_files[:limit]
            image_files = image_files[:limit]
        
        # Process videos
        if video_files:
            print(f"Processing {len(video_files)} videos...")
            for video_file in tqdm(video_files):
                video_path = os.path.join(input_dir, video_file)
                media_id = os.path.splitext(video_file)[0]
                
                embedding, frames, processing_time = self.generate_video_embedding(video_path)
                
                if embedding is not None:
                    # Save the embedding
                    embedding_file = os.path.join(output_dir, f"{media_id}_embedding.npy")
                    np.save(embedding_file, embedding)
                    
                    # Save metadata
                    file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
                    
                    metadata = {
                        "media_id": media_id,
                        "media_type": "video",
                        "timestamp": datetime.now().isoformat(),
                        "embedding_file": embedding_file,
                        "embedding_dimension": embedding.shape[0],
                        "original_file": video_path,
                        "file_size_mb": round(file_size, 2),
                        "num_frames_processed": len(frames),
                        "processing_time_seconds": round(processing_time, 3),
                        "embedding_version": OUTPUT_VERSION,
                        "model_used": "resnet18"
                    }
                    
                    metadata_file = os.path.join(output_dir, f"{media_id}_metadata.json")
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    
                    results.append(metadata)
        
        # Process images
        if image_files:
            print(f"Processing {len(image_files)} images...")
            for image_file in tqdm(image_files):
                image_path = os.path.join(input_dir, image_file)
                media_id = os.path.splitext(image_file)[0]
                
                embedding, processing_time = self.generate_image_embedding(image_path)
                
                if embedding is not None:
                    # Save the embedding
                    embedding_file = os.path.join(output_dir, f"{media_id}_embedding.npy")
                    np.save(embedding_file, embedding)
                    
                    # Save metadata
                    file_size = os.path.getsize(image_path) / (1024 * 1024)  # Size in MB
                    
                    metadata = {
                        "media_id": media_id,
                        "media_type": "image",
                        "timestamp": datetime.now().isoformat(),
                        "embedding_file": embedding_file,
                        "embedding_dimension": embedding.shape[0],
                        "original_file": image_path,
                        "file_size_mb": round(file_size, 2),
                        "processing_time_seconds": round(processing_time, 3),
                        "embedding_version": OUTPUT_VERSION,
                        "model_used": "resnet18"
                    }
                    
                    metadata_file = os.path.join(output_dir, f"{media_id}_metadata.json")
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    
                    results.append(metadata)
        
        return results
    
    def generate_batch_embeddings(self, input_dirs, output_dir, limit=None):
        """Generate embeddings for all media files in multiple directories."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Split input_dirs into a list if it's a comma-separated string
        if isinstance(input_dirs, str):
            input_dirs = [d.strip() for d in input_dirs.split(',')]
        
        all_results = []
        
        # Process each directory
        for input_dir in input_dirs:
            print(f"\nProcessing directory: {input_dir}")
            results = self.process_directory(input_dir, output_dir, limit)
            all_results.extend(results)
        
        # Save batch summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_media_processed": len(all_results),
            "num_videos_processed": sum(1 for r in all_results if r["media_type"] == "video"),
            "num_images_processed": sum(1 for r in all_results if r["media_type"] == "image"),
            "average_processing_time": round(np.mean([r["processing_time_seconds"] for r in all_results]), 3) if all_results else 0,
            "total_processing_time": round(sum([r["processing_time_seconds"] for r in all_results]), 3),
            "directories_processed": input_dirs,
            "media_processed": [r["media_id"] for r in all_results],
            "embedding_version": OUTPUT_VERSION
        }
        
        summary_file = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nSummary:")
        print(f"Processed {summary['num_videos_processed']} videos and {summary['num_images_processed']} images")
        print(f"Total processing time: {summary['total_processing_time']} seconds")
        print(f"Average processing time: {summary['average_processing_time']} seconds per file")
        print(f"Embeddings saved to {output_dir}")
        
        return all_results

# Alias for backward compatibility
VideoEmbeddingGenerator = MediaEmbeddingGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for videos and images for similarity search")
    parser.add_argument("--input", required=True, help="Input directory or comma-separated list of directories containing media files")
    parser.add_argument("--output", required=True, help="Output directory for embeddings")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"], 
                        help="Model to use for feature extraction")
    parser.add_argument("--limit", type=int, help="Limit number of files to process per directory")
    
    args = parser.parse_args()
    
    generator = MediaEmbeddingGenerator(model_name=args.model)
    generator.generate_batch_embeddings(args.input, args.output, limit=args.limit)

if __name__ == "__main__":
    main() 