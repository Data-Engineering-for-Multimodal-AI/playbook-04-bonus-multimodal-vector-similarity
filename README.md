# ByteEats Vector Similarity Search

An advanced vector embedding and similarity search system for food preparation videos. This system demonstrates complex data transformation by converting unstructured video data into structured vector embeddings and implementing efficient similarity search.

## Features

- **Video Feature Extraction**: Extracts meaningful features from food preparation videos
- **Vector Embedding Generation**: Transforms video features into 512-dimensional vector embeddings using ResNet
- **Similarity Search**: Efficiently finds similar food preparations using vector similarity with FAISS
- **Quality Assessment**: Compares videos against reference images with configurable thresholds
- **Visualization**: Interactive dashboard to explore embeddings and quality assessment results

## Bonus: Scaling Similarity Search Guide

For a comprehensive, step-by-step walkthrough of this entire system — from embedding generation through quality assessment to production scaling — see the **[Bonus Content Guide](docs/GUIDE.md)**.

The guide includes:
- Detailed explanations of each pipeline stage with expected outputs
- Dashboard screenshots showing PCA visualizations and quality assessments
- Side-by-side comparisons of video frames vs. reference images
- Practical advice on improving matching quality (fine-tuning, dynamic thresholds, weighted similarity)
- Production scaling strategies (vector databases, microservices, incremental processing)

> **Requires Python >= 3.10.** See [requirements.txt](requirements.txt) for dependencies.

## Directory Structure

```
ByteEatsVectorSimilaritySearch/
├── src/                    # Source code
│   ├── embeddings/         # Vector embedding generation
│   ├── similarity/         # Similarity search algorithms
│   ├── api/                # FastAPI endpoints and dashboard
│   └── utils/              # Utility functions including quality_checker
├── data/                   # Sample data
│   ├── videos/             # Food preparation videos
│   ├── images/             # Reference food images
│   ├── embeddings/         # Generated vector embeddings
│   └── quality/            # Quality assessment reports
├── scripts/                # Utility scripts for setup and cleanup
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ByteEatsVectorSimilaritySearch.git
cd ByteEatsVectorSimilaritySearch

# Use the provided installation script
chmod +x install.sh
./install.sh

# Or manually install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Detailed Implementation

### 1. Embedding Generation

The embedding generation process converts videos and images into 512-dimensional vectors using a pre-trained ResNet model:

```bash
export IMAGE_DIR=data/images
./run_pipeline.sh --generate-embeddings
```

#### How it Works:

1. **Video Processing**:
   - Loads videos from the specified directory
   - Extracts frames at regular intervals (every 5th frame by default)
   - Processes each frame through ResNet to extract features
   - Averages the frame embeddings to create a single video embedding
   - Normalizes the embeddings to ensure consistent similarity calculations

2. **Image Processing**:
   - Loads reference images from the images directory
   - Processes each image through the same ResNet model
   - Normalizes the embeddings for consistent comparison with video embeddings

3. **Output**:
   - Each embedding is saved as a .npy file in the data/embeddings directory
   - Metadata is saved alongside each embedding in JSON format
   - For videos: includes source path, duration, frame count, and processing time
   - For images: includes source path, resolution, and processing time

#### Technical Details:

- Uses transfer learning with ResNet-18 pretrained on ImageNet
- Removes the classification layer to extract 512-dimensional feature vectors
- Applies L2 normalization to embeddings for optimal cosine similarity calculation
- Processes both images and videos through the same pipeline for compatibility

### 2. Similarity Search API

The API server provides endpoints for searching similar videos and performing quality checks:

```bash
export API_PORT=8085
export IMAGE_DIR=data/images
./run_pipeline.sh --start-api
```

#### API Architecture:

1. **FastAPI Backend**:
   - RESTful API with documented endpoints (Swagger available at /docs)
   - Handles requests for similarity search and quality assessment
   - Maintains a in-memory FAISS index for fast similarity queries
   - Processes quality check requests by comparing videos to reference images

2. **FAISS Index**:
   - Creates a high-performance vector similarity search index
   - Supports different index types (Flat, IVF, HNSW)
   - Optimized for fast nearest-neighbor search in high-dimensional space
   - Automatically rebuilt when new embeddings are detected

3. **Quality Check**:
   - Loads video frames and reference images
   - Computes frame-by-frame similarity scores
   - Determines quality based on configurable thresholds
   - Generates detailed reports saved to the quality directory

#### API Endpoints:

- `GET /api/videos`: List all available videos
- `GET /api/quality/reference_images`: List all reference images
- `POST /api/quality/check`: Check quality of a video against a reference image
- `GET /api/quality/reports`: Get all quality reports
- `GET /api/similar/{video_id}`: Find videos similar to a specific video
- `POST /api/search`: Search for videos using a custom query

### 3. Dashboard Visualization

The dashboard provides an interactive interface to explore embeddings and assess quality:

```bash
export API_PORT=8085
export DASHBOARD_PORT=8095
export API_URL="http://localhost:8085"
./run_pipeline.sh --start-dashboard
```

#### Dashboard Features:

1. **Vector Embedding Visualization**:
   - Interactive 2D plot using PCA (Principal Component Analysis)
   - Visualizes relationships between videos and reference images
   - Color-coded data points for easy identification
   - Hover tooltips with detailed information

2. **Quality Assessment Interface**:
   - Select videos and reference images from dropdown menus
   - Configure quality threshold with a slider (0.5-0.95)
   - View side-by-side comparison of video frames and reference images
   - See detailed similarity scores and assessment results

3. **Similarity Search**:
   - Find videos similar to a selected video
   - View ranked results with similarity scores
   - Visualize embedding clusters and patterns

#### Technical Implementation:

- Built with Dash and Plotly for interactive visualizations
- Communicates with the API server using HTTP requests
- Updates visualizations in real-time as new data becomes available
- Responsive design for different screen sizes

### 4. Quality Check Tool

The quality check tool provides a command-line interface for assessing video quality:

```bash
export API_PORT=8085
export IMAGE_DIR=data/images
./run_pipeline.sh --quality-check
```

#### Quality Assessment Process:

1. **Initialization**:
   - Loads available videos and reference images
   - Prompts user to select a video, reference image, and threshold

2. **Assessment**:
   - Extracts frames from the selected video
   - Generates embeddings for video frames
   - Compares each frame to the reference image embedding
   - Finds the best matching frame and its similarity score

3. **Result Determination**:
   - PASSED: Similarity score ≥ threshold (good quality match)
   - FAILED: Similarity score < threshold (poor quality match)
   - Quality reports include detailed frame-by-frame similarity metrics

4. **Output**:
   - Detailed assessment with similarity scores and threshold information
   - Processing time and technical details
   - Option to view all quality reports in a tabular format

#### Similarity Score Interpretation:

- 0.75-0.85: Excellent match (very similar preparation)
- 0.65-0.75: Good match (similar preparation with minor differences)
- 0.55-0.65: Fair match (recognizable but with significant differences)
- Below 0.55: Poor match (different preparation or significant issues)

## Troubleshooting

### Common Issues:

1. **Port Conflicts**:
   - If you encounter "address already in use" errors, change the port:
   ```bash
   export API_PORT=8086  # Use a different port number
   ```

2. **Image Directory Not Found**:
   - Ensure the IMAGE_DIR environment variable points to the correct path:
   ```bash
   export IMAGE_DIR=data/images
   ```

3. **No Embeddings Found**:
   - Run the embedding generation step first:
   ```bash
   ./run_pipeline.sh --generate-embeddings
   ```

4. **Dashboard Connection Issues**:
   - Make sure the API server is running
   - Verify the API_URL points to the correct address and port:
   ```bash
   export API_URL="http://localhost:8085"
   ```

## Performance Considerations

For optimal performance when scaling to larger datasets:

1. **Embedding Generation**:
   - Processing thousands of videos requires distributed computation
   - Consider using GPU acceleration for 10-20x speedup
   - Batch processing for efficient resource utilization

2. **Similarity Search**:
   - FAISS indexes can be configured for different performance profiles:
     - Flat: Most accurate but slower for large collections
     - IVF: Faster but slightly less accurate
     - HNSW: Best balance of speed and accuracy for large datasets

3. **Memory Management**:
   - For large datasets, consider memory-mapped storage of embeddings
   - Implement pagination for API results
   - Use quantization techniques to reduce embedding size (FP16, INT8)

## Data Engineering Highlights

This project showcases advanced data engineering techniques:

- **Complex Data Transformation**: Converting raw video data into structured vector embeddings
- **Transfer Learning**: Using pre-trained ResNet models to extract visual features
- **Temporal Aggregation**: Averaging frame embeddings for consistent video representation
- **Fast Similarity Search**: Using FAISS for efficient nearest-neighbor queries
- **Cross-Modal Comparison**: Matching videos against images using the same embedding space

## License

The ByteEats Vector Similarity Search project is licensed under the MIT License.
