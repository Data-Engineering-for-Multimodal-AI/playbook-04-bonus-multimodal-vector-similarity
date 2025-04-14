#!/usr/bin/env python3
"""
Dashboard for ByteEats Vector Similarity Search

This module provides a web dashboard for visualizing video embeddings
and similarity search results.
"""

import os
import sys
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import requests

# Add parent directory to path to import modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

# Define the API URL
API_URL = os.environ.get("API_URL", "http://localhost:8080")

# Initialize the Dash app
app = Dash(__name__, title="ByteEats Embedding Dashboard")

# Define the app layout
app.layout = html.Div([
    html.Div([
        html.H1("ByteEats Vector Similarity Search Dashboard"),
        html.P("Visualize video embeddings and find similar food preparations")
    ], style={'textAlign': 'center', 'margin-bottom': '20px'}),
    
    # Add tabs for different sections
    dcc.Tabs([
        # First tab - Vector Embedding & Similarity
        dcc.Tab(label="Vector Embedding & Similarity", children=[
            html.Div([
                html.Div([
                    html.H3("Embedding Visualization"),
                    dcc.Dropdown(
                        id='projection-method',
                        options=[
                            {'label': 'PCA (Principal Component Analysis)', 'value': 'pca'},
                            {'label': 't-SNE (t-Distributed Stochastic Neighbor Embedding)', 'value': 'tsne'}
                        ],
                        value='pca',
                        style={'width': '100%'}
                    ),
                    dcc.Graph(id='embedding-plot', style={'height': '600px'})
                ], className='six columns', style={'width': '48%'}),
                
                html.Div([
                    html.H3("Similarity Search"),
                    html.Div([
                        html.Label("Select a video:"),
                        dcc.Dropdown(id='video-dropdown', style={'width': '100%'}),
                        html.Button('Find Similar', id='search-button', n_clicks=0, 
                                style={'margin-top': '10px', 'margin-bottom': '20px'})
                    ]),
                    
                    html.Div(id='search-results', style={'margin-top': '20px'})
                ], className='six columns', style={'width': '48%', 'float': 'right'})
            ], className='row'),
            
            html.Div([
                html.H3("Cluster Analysis"),
                html.Button('Analyze Clusters', id='analyze-button', n_clicks=0),
                html.Div(id='cluster-results', style={'margin-top': '20px'})
            ], style={'margin-top': '40px', 'clear': 'both'}),
        ]),
        
        # New tab - Quality Assessment
        dcc.Tab(label="Quality Assessment", children=[
            html.Div([
                html.H3("Food Quality Check"),
                html.Div([
                    html.Div([
                        html.Label("Select Video:"),
                        dcc.Dropdown(id='quality-video-dropdown', style={'width': '100%', 'marginBottom': '15px'}),
                        html.Label("Select Reference Image:"),
                        dcc.Dropdown(id='reference-image-dropdown', style={'width': '100%', 'marginBottom': '15px'}),
                        html.Label("Quality Threshold:"),
                        dcc.Slider(
                            id='quality-threshold-slider',
                            min=0.5,
                            max=0.95,
                            step=0.05,
                            value=0.75,
                            marks={i/100: {'label': str(i/100)} for i in range(50, 100, 5)},
                            tooltip={'placement': 'bottom', 'always_visible': True}
                        ),
                        html.Button('Check Quality', id='quality-check-button', n_clicks=0, 
                                style={'margin-top': '20px', 'marginBottom': '20px', 'width': '100%'})
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div(id='quality-results', style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
                ]),
                
                html.Div([
                    html.H3("Quality Reports"),
                    html.Button('Load Reports', id='load-reports-button', n_clicks=0,
                              style={'marginBottom': '15px'}),
                    html.Div(id='quality-reports-table')
                ], style={'marginTop': '40px'})
            ])
        ])
    ]),
    
    html.Div([
        html.H3("System Statistics"),
        html.Button('Refresh Stats', id='stats-button', n_clicks=0),
        html.Div(id='stats-results', style={'margin-top': '20px'})
    ], style={'margin-top': '40px'})
])

# Load videos on page load
@app.callback(
    Output('video-dropdown', 'options'),
    Input('video-dropdown', 'id')
)
def load_videos(_):
    try:
        response = requests.get(f"{API_URL}/api/videos")
        data = response.json()
        
        if response.status_code == 200:
            # The API returns a list of video objects directly
            videos = data
            # Filter for only video media type
            video_entries = [v.get('video_id') for v in videos if v.get('media_type') == 'video']
            return [{'label': video_id, 'value': video_id} for video_id in video_entries]
        else:
            return []
    except Exception as e:
        print(f"Error loading videos: {str(e)}")
        return []

# Generate the embedding plot
@app.callback(
    Output('embedding-plot', 'figure'),
    [Input('projection-method', 'value')]
)
def update_embedding_plot(projection_method):
    try:
        # Fetch embeddings from API (for a real dashboard, you'd call an API endpoint)
        # For demonstration, we'll load from files directly
        embedding_dir = os.path.join(parent_dir, "data/embeddings")
        
        embeddings = []
        video_ids = []
        
        embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('_embedding.npy')]
        
        for embedding_file in embedding_files:
            video_id = embedding_file.replace('_embedding.npy', '')
            embedding_path = os.path.join(embedding_dir, embedding_file)
            
            try:
                embedding = np.load(embedding_path)
                embeddings.append(embedding)
                video_ids.append(video_id)
            except Exception as e:
                print(f"Error loading embedding {embedding_file}: {str(e)}")
        
        if not embeddings:
            return go.Figure().update_layout(
                title="No embeddings available",
                xaxis_title="X",
                yaxis_title="Y"
            )
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Apply dimensionality reduction
        if projection_method == 'pca':
            pca = PCA(n_components=2)
            projected = pca.fit_transform(embeddings)
            explained_var = pca.explained_variance_ratio_
            title = f"PCA Projection (Explained variance: {explained_var[0]:.2f}, {explained_var[1]:.2f})"
        else:  # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            projected = tsne.fit_transform(embeddings)
            title = "t-SNE Projection"
        
        # Create the plot
        fig = px.scatter(
            x=projected[:, 0],
            y=projected[:, 1],
            text=video_ids,
            hover_data={"ID": video_ids},
            title=title
        )
        
        fig.update_traces(
            marker=dict(size=10, opacity=0.8),
            textposition="top center"
        )
        
        fig.update_layout(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            xaxis=dict(zeroline=False),
            yaxis=dict(zeroline=False)
        )
        
        return fig
    
    except Exception as e:
        print(f"Error generating embedding plot: {str(e)}")
        return go.Figure().update_layout(
            title=f"Error: {str(e)}",
            xaxis_title="X",
            yaxis_title="Y"
        )

# Search for similar videos
@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks')],
    [State('video-dropdown', 'value')]
)
def search_similar(n_clicks, video_id):
    if n_clicks == 0 or not video_id:
        return html.Div("Select a video and click 'Find Similar' to see results.")
    
    try:
        # Make API request
        response = requests.post(
            f"{API_URL}/api/similarity/search",
            json={"video_id": video_id, "k": 5}
        )
        
        if response.status_code != 200:
            return html.Div(f"Error: {response.text}")
        
        data = response.json()
        results = data.get('results', [])
        stats = data.get('stats', {})
        
        if not results:
            return html.Div("No similar videos found.")
        
        # Create results display
        result_elements = [
            html.Div([
                html.H4(f"Query: {video_id}"),
                html.P(f"Search time: {stats.get('search_time_ms', 0)} ms")
            ])
        ]
        
        result_elements.append(html.Hr())
        
        for result in results:
            result_elements.append(html.Div([
                html.H5(f"{result['rank']}. {result['video_id']}"),
                html.P(f"Similarity score: {result['similarity_score']:.3f}"),
                html.P(f"Distance: {result['distance']:.3f}")
            ], style={'margin-bottom': '10px', 'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px'}))
        
        return html.Div(result_elements)
    
    except Exception as e:
        return html.Div(f"Error: {str(e)}")

# Analyze clusters
@app.callback(
    Output('cluster-results', 'children'),
    [Input('analyze-button', 'n_clicks')]
)
def analyze_clusters(n_clicks):
    if n_clicks == 0:
        return html.Div("Click 'Analyze Clusters' to perform cluster analysis.")
    
    try:
        # Make API request
        response = requests.get(f"{API_URL}/api/similarity/analyze")
        
        if response.status_code != 200:
            return html.Div(f"Error: {response.text}")
        
        data = response.json()
        clusters = data.get('clusters', [])
        outliers = data.get('outliers', [])
        
        # Create cluster display
        result_elements = [
            html.Div([
                html.H4(f"Found {len(clusters)} clusters and {len(outliers)} outliers"),
                html.P(f"Analysis time: {data.get('analysis_time_seconds', 0)} seconds")
            ])
        ]
        
        # Show clusters
        result_elements.append(html.H5("Clusters:"))
        
        for cluster in clusters:
            videos = ", ".join(cluster['videos'][:5])
            if len(cluster['videos']) > 5:
                videos += f" and {len(cluster['videos']) - 5} more"
            
            result_elements.append(html.Div([
                html.P(f"Cluster {cluster['cluster_id']}: {cluster['size']} videos, cohesion: {cluster['cohesion']:.3f}"),
                html.P(f"Videos: {videos}")
            ], style={'margin-bottom': '10px', 'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px'}))
        
        # Show outliers
        result_elements.append(html.H5("Top Outliers:"))
        
        for outlier in outliers[:5]:
            result_elements.append(html.Div([
                html.P(f"{outlier['video_id']} - Isolation score: {outlier['isolation_score']:.3f}")
            ], style={'margin-bottom': '10px', 'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px'}))
        
        return html.Div(result_elements)
    
    except Exception as e:
        return html.Div(f"Error: {str(e)}")

# Show system statistics
@app.callback(
    Output('stats-results', 'children'),
    [Input('stats-button', 'n_clicks')]
)
def show_stats(n_clicks):
    if n_clicks == 0:
        return html.Div("Click 'Refresh Stats' to view system statistics.")
    
    try:
        # Make API request
        response = requests.get(f"{API_URL}/api/stats")
        
        if response.status_code != 200:
            return html.Div(f"Error: {response.text}")
        
        data = response.json()
        
        # Create stats display
        result_elements = [
            html.Div([
                html.H4("System Statistics"),
                html.P(f"Number of videos: {data.get('num_videos', 0)}"),
                html.P(f"Embedding dimension: {data.get('embedding_dimension', 0)}"),
                html.P(f"Total storage: {data.get('total_storage_mb', 0)} MB"),
                html.P(f"Average video size: {data.get('average_video_size_mb', 0)} MB"),
                html.P(f"Embedding to video ratio: {data.get('embedding_to_video_ratio', 0)}"),
                html.P(f"Timestamp: {data.get('timestamp', '')}")
            ], style={'padding': '15px', 'border': '1px solid #ddd', 'border-radius': '5px'})
        ]
        
        return html.Div(result_elements)
    
    except Exception as e:
        return html.Div(f"Error: {str(e)}")

# Load reference images
@app.callback(
    Output('reference-image-dropdown', 'options'),
    Input('reference-image-dropdown', 'id')
)
def load_reference_images(_):
    try:
        response = requests.get(f"{API_URL}/api/quality/reference_images")
        data = response.json()
        
        if response.status_code == 200:
            # API returns a direct list of image filenames
            images = data.get('images', [])
            return [{'label': image, 'value': image} for image in images]
        else:
            return []
    except Exception as e:
        print(f"Error loading reference images: {str(e)}")
        return []

# Populate quality video dropdown with same videos as similarity search
@app.callback(
    Output('quality-video-dropdown', 'options'),
    Input('video-dropdown', 'options')
)
def update_quality_video_dropdown(options):
    return options

# Quality check button callback
@app.callback(
    Output('quality-results', 'children'),
    [Input('quality-check-button', 'n_clicks')],
    [State('quality-video-dropdown', 'value'),
     State('reference-image-dropdown', 'value'),
     State('quality-threshold-slider', 'value')]
)
def check_quality(n_clicks, video_id, reference_image, threshold):
    if n_clicks == 0 or not video_id or not reference_image:
        return html.Div("Select a video and reference image, then click 'Check Quality' to assess food quality.")
    
    try:
        # Make API request with the correct format
        response = requests.post(
            f"{API_URL}/api/quality/check",
            json={"video_id": video_id, "reference_image": reference_image, "threshold": threshold}
        )
        
        if response.status_code != 200:
            return html.Div(f"Error: {response.text}")
        
        report = response.json()
        
        # Create quality report display
        quality_status = report.get('quality_passed', False)
        # Use either similarity_score or max_similarity_score depending on what's available
        similarity_score = report.get('similarity_score', report.get('max_similarity_score', 0))
        
        status_color = 'green' if quality_status else 'red'
        status_text = 'PASSED' if quality_status else 'FAILED'
        
        result_elements = [
            html.Div([
                html.H4("Quality Assessment Results", style={'textAlign': 'center'}),
                html.Div([
                    html.H2(status_text, style={'color': status_color, 'textAlign': 'center'})
                ], style={'padding': '10px', 'border': f'2px solid {status_color}', 'borderRadius': '5px', 'marginBottom': '20px'}),
                html.Hr(),
                html.Div([
                    html.P(f"Video: {report.get('video_id', '')}"),
                    html.P(f"Reference Image: {report.get('reference_image', '')}"),
                    html.P(f"Similarity Score: {similarity_score:.4f}", style={'fontWeight': 'bold'}),
                    html.P(f"Threshold: {report.get('threshold', 0):.2f}"),
                    html.P(f"Processing Time: {report.get('processing_time_seconds', 0):.3f} seconds"),
                    html.P(f"Assessment: {report.get('assessment', '')}", style={'fontWeight': 'bold'}),
                    html.P(f"Timestamp: {report.get('timestamp', '')}")
                ], style={'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px'})
            ])
        ]
        
        return html.Div(result_elements)
    
    except Exception as e:
        return html.Div(f"Error: {str(e)}")

# Load quality reports
@app.callback(
    Output('quality-reports-table', 'children'),
    [Input('load-reports-button', 'n_clicks')]
)
def load_quality_reports(n_clicks):
    if n_clicks == 0:
        return html.Div("Click 'Load Reports' to view quality assessment reports.")
    
    try:
        # Make API request
        response = requests.get(f"{API_URL}/api/quality/reports")
        
        if response.status_code != 200:
            return html.Div(f"Error: {response.text}")
        
        data = response.json()
        reports = data.get('reports', [])
        
        if not reports:
            return html.Div("No quality reports found.")
        
        # Create table header
        header = html.Thead(html.Tr([
            html.Th("Video ID"),
            html.Th("Reference Image"),
            html.Th("Similarity Score"),
            html.Th("Threshold"),
            html.Th("Status"),
            html.Th("Timestamp")
        ]))
        
        # Create table rows
        rows = []
        for report in reports:
            quality_passed = report.get('quality_passed', False)
            status_style = {'color': 'green', 'fontWeight': 'bold'} if quality_passed else {'color': 'red', 'fontWeight': 'bold'}
            
            # Use either similarity_score or max_similarity_score
            similarity_score = report.get('similarity_score', report.get('max_similarity_score', 0))
            
            rows.append(html.Tr([
                html.Td(report.get('video_id', '')),
                html.Td(report.get('reference_image', '')),
                html.Td(f"{similarity_score:.4f}"),
                html.Td(f"{report.get('threshold', 0):.2f}"),
                html.Td("PASSED" if quality_passed else "FAILED", style=status_style),
                html.Td(report.get('timestamp', '').split('T')[0])
            ]))
        
        table_body = html.Tbody(rows)
        
        # Assemble table
        table = html.Table([header, table_body], style={
            'width': '100%',
            'border': '1px solid #ddd',
            'borderCollapse': 'collapse'
        })
        
        return table
    
    except Exception as e:
        return html.Div(f"Error: {str(e)}")

def main():
    """Run the dashboard server."""
    # Determine port (use different port than API)
    port = int(os.environ.get("DASHBOARD_PORT", 8050))
    
    print(f"Starting ByteEats Vector Similarity Search Dashboard on port {port}")
    print(f"API URL: {API_URL}")
    
    app.run(host="0.0.0.0", port=port, debug=True)

if __name__ == "__main__":
    main() 