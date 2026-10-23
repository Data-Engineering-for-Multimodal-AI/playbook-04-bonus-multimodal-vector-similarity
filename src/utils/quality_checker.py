#!/usr/bin/env python3
"""
Quality Checker for ByteEats Vector Similarity Search

This module compares reference food images against preparation videos
to assess quality and detect deviations.
"""

import os
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import torchvision.transforms as transforms
from datetime import datetime
import json
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

class QualityChecker:
    def __init__(self, model_name="resnet18", device=None):
        """Initialize the quality checker with a pre-trained model."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load pre-trained model (the same ImageNet weights pretrained=True loaded; that argument is deprecated)
        if model_name == "resnet18":
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            # Remove the final classification layer
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        elif model_name == "resnet50":
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_frames(self, video_path, num_frames=8):
        """Extract frames from a video file."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        
        # Calculate frames to sample
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames, duration
    
    def load_reference_image(self, image_path):
        """Load a reference image and convert to RGB."""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading reference image: {str(e)}")
            return None
    
    def generate_embedding(self, image):
        """Generate an embedding for an image."""
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image)
        
        with torch.no_grad():
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            feature = self.model(input_tensor)
            embedding = feature.squeeze().cpu().numpy()
            
            # Normalize the embedding vector
            embedding = embedding / np.linalg.norm(embedding)
            
        return embedding
    
    def compare_images(self, reference_image, frame):
        """Compare a reference image to a video frame."""
        ref_embedding = self.generate_embedding(reference_image)
        frame_embedding = self.generate_embedding(frame)
        
        # Calculate cosine similarity
        similarity = cosine_similarity([ref_embedding], [frame_embedding])[0][0]
        
        return similarity
    
    def assess_quality(self, video_path, reference_image_path, threshold=0.75):
        """Assess the quality of a food preparation video by comparing to a reference image."""
        reference_image = self.load_reference_image(reference_image_path)
        if reference_image is None:
            return None
        
        frames, duration = self.extract_frames(video_path)
        if not frames:
            return None
        
        # Compare each frame to the reference image
        similarities = []
        for frame in frames:
            similarity = self.compare_images(reference_image, frame)
            # Convert numpy float to Python float to ensure JSON serialization
            similarities.append(float(similarity))
        
        # Find the frame with the highest similarity
        max_similarity = max(similarities)
        best_frame_index = similarities.index(max_similarity)
        
        # Quality assessment
        quality_passed = max_similarity >= threshold
        
        # Create quality report with all values JSON serializable
        report = {
            "video_id": os.path.splitext(os.path.basename(video_path))[0],
            "reference_image": os.path.basename(reference_image_path),
            "timestamp": datetime.now().isoformat(),
            "quality_passed": bool(quality_passed),  # Ensure boolean is native Python bool
            "max_similarity_score": float(max_similarity),  # Ensure float is native Python float
            "threshold": float(threshold),  # Ensure float is native Python float
            "best_frame_index": int(best_frame_index),  # Ensure int is native Python int
            "frame_similarities": similarities,  # Already converted to native Python floats
            "assessment": "Acceptable" if quality_passed else "Needs review"
        }
        
        return report
    
    def batch_quality_check(self, video_dir, image_dir, output_dir, threshold=0.75):
        """Run quality checks on multiple videos against reference images."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all video files
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        # Get all reference images
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not video_files or not image_files:
            print("No videos or reference images found.")
            return []
        
        # Use the first reference image for now
        # In a real system, you would match videos to the correct reference image
        reference_image_path = os.path.join(image_dir, image_files[0])
        
        reports = []
        
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            video_id = os.path.splitext(video_file)[0]
            
            print(f"Assessing quality for video: {video_id}")
            
            report = self.assess_quality(video_path, reference_image_path, threshold)
            if report:
                # Save the report
                report_file = os.path.join(output_dir, f"{video_id}_quality.json")
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=4)
                
                reports.append(report)
        
        # Save batch summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "videos_assessed": len(reports),
            "videos_passed": sum(1 for r in reports if r["quality_passed"]),
            "videos_failed": sum(1 for r in reports if not r["quality_passed"]),
            "average_similarity": sum(r["max_similarity_score"] for r in reports) / len(reports) if reports else 0,
            "threshold": threshold
        }
        
        summary_file = os.path.join(output_dir, f"quality_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Quality assessment complete. {summary['videos_passed']} passed, {summary['videos_failed']} need review.")
        
        return reports

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Quality checker for food preparation videos")
    parser.add_argument("--video", help="Path to a video file for quality assessment")
    parser.add_argument("--reference", help="Path to a reference image for comparison")
    parser.add_argument("--video-dir", help="Directory containing video files")
    parser.add_argument("--image-dir", help="Directory containing reference images")
    parser.add_argument("--output", required=True, help="Output directory for quality reports")
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold for quality passing")
    
    args = parser.parse_args()
    
    checker = QualityChecker()
    
    if args.video and args.reference:
        # Assess a single video
        report = checker.assess_quality(args.video, args.reference, args.threshold)
        if report:
            report_file = os.path.join(args.output, f"{report['video_id']}_quality.json")
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)
            
            print(f"Quality assessment complete: {report['assessment']}")
            print(f"Similarity score: {report['max_similarity_score']:.4f} (threshold: {args.threshold})")
            print(f"Report saved to {report_file}")
    
    elif args.video_dir and args.image_dir:
        # Assess multiple videos
        checker.batch_quality_check(args.video_dir, args.image_dir, args.output, args.threshold)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 