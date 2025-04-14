# ByteEats Vector Similarity Search

An advanced vector embedding and similarity search system for food preparation videos. This system demonstrates complex data transformation by converting unstructured video data into structured vector embeddings and implementing efficient similarity search.

## Features

- **Video Feature Extraction**: Extracts meaningful features from food preparation videos
- **Vector Embedding Generation**: Transforms video features into high-dimensional vector embeddings
- **Similarity Search**: Efficiently finds similar food preparations using vector similarity
- **Performance Metrics**: Tracks embedding generation time and query latency
- **Visualization**: Interactive dashboard to explore embeddings and similarity results

## Directory Structure

```
ByteEatsVectorSimilaritySearch/
├── src/                    # Source code
│   ├── embeddings/         # Vector embedding generation
│   ├── similarity/         # Similarity search algorithms
│   ├── api/                # FastAPI endpoints
│   └── utils/              # Utility functions
├── data/                   # Sample data
│   ├── videos/             # Food preparation videos
│   ├── images/             # Reference food images
│   ├── embeddings/         # Generated vector embeddings
│   └── results/            # Similarity search results
├── models/                 # Pre-trained models and weights
├── notebooks/              # Jupyter notebooks for experimentation
├── docs/                   # Documentation
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ByteEatsVectorSimilaritySearch.git
cd ByteEatsVectorSimilaritySearch

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generate Embeddings

```bash
python src/embeddings/generator.py --input data/videos --output data/embeddings
```

### Run Similarity Search API

```bash
python src/api/main.py
```

### Launch Dashboard

```bash
python src/api/dashboard.py
```

## Sample Data

To demonstrate the system, you'll need:
1. 50+ short food preparation videos (5-8 seconds each)
2. 8+ reference images of perfectly prepared dishes

## Data Engineering Highlights

This project showcases advanced data engineering techniques:

- **Complex Data Transformation**: Converting raw video data into structured vector embeddings
- **Scalability**: Processing multiple videos in parallel
- **Storage Efficiency**: Comparing raw video size to embedding size
- **Query Optimization**: Fast similarity search using FAISS

## License

MIT 