#!/usr/bin/env python3
"""
Cleanup Script for ByteEatsVectorSimilaritySearch

This script cleans up temporary files, test artifacts, and other
files that shouldn't be included in the GitHub repository.
"""

import os
import sys
import argparse
import shutil

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

def clean_pycache():
    """Remove all __pycache__ directories."""
    count = 0
    for root, dirs, files in os.walk(parent_dir):
        if "__pycache__" in dirs:
            pycache_dir = os.path.join(root, "__pycache__")
            shutil.rmtree(pycache_dir)
            count += 1
    
    if count > 0:
        print(f"Removed {count} __pycache__ directories")
    else:
        print("No __pycache__ directories found")

def clean_quality_reports():
    """Clean up quality report files, but leave one example."""
    quality_dir = os.path.join(parent_dir, "data", "quality")
    if not os.path.exists(quality_dir):
        print("No quality reports directory found")
        return
    
    # Get all reports
    reports = [f for f in os.listdir(quality_dir) if f.endswith('_quality.json')]
    
    if not reports:
        print("No quality reports found")
        return
    
    # Keep just the first one as an example
    reports_to_delete = reports[1:]
    
    if not reports_to_delete:
        print("Only one quality report found, keeping it as an example")
        return
    
    for report in reports_to_delete:
        os.remove(os.path.join(quality_dir, report))
    
    print(f"Removed {len(reports_to_delete)} quality reports, kept 1 as an example")

def clean_temp_files():
    """Clean up temporary files."""
    extensions = ['.log', '.tmp', '.bak', '.swp', '.DS_Store']
    count = 0
    
    for root, dirs, files in os.walk(parent_dir):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                count += 1
    
    if count > 0:
        print(f"Removed {count} temporary files")
    else:
        print("No temporary files found")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ByteEats Cleanup Tool")
    parser.add_argument("--pycache", action="store_true", help="Clean __pycache__ directories")
    parser.add_argument("--reports", action="store_true", help="Clean quality reports")
    parser.add_argument("--temp", action="store_true", help="Clean temporary files")
    parser.add_argument("--all", action="store_true", help="Clean everything")
    
    args = parser.parse_args()
    
    if args.pycache or args.all:
        clean_pycache()
    
    if args.reports or args.all:
        clean_quality_reports()
    
    if args.temp or args.all:
        clean_temp_files()
    
    if not (args.pycache or args.reports or args.temp or args.all):
        print("No cleanup options specified. Use --all to clean everything.")
        parser.print_help()

if __name__ == "__main__":
    main() 