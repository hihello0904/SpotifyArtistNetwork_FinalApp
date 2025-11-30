# Spotify Artist Network

An interactive Python Shiny app that visualizes Spotify artist collaboration networks with filters and multiple visualizations.

## Prerequisites

- Python 3.9+
- pip

## Setup

1. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

With the virtual environment activated, run:

```bash
shiny run spotify_shiny_app.py
```

The app will start and be available at **http://127.0.0.1:8000**

> **Note:** The first startup may take a few minutes as it computes network centrality metrics and graph layout for ~10,000 nodes.

## Features

- **Collaboration Network**: Interactive network graph showing artist collaborations
- **Degree vs Popularity**: Scatter plot of collaborations vs Spotify popularity
- **Popularity vs Followers**: Scatter plot with logarithmic follower scale
- **Degree Distribution**: Histogram of collaboration counts

## Data Files

- `nodes.csv` - Artist metadata (spotify_id, name, followers, popularity)
- `edges.csv` - Collaboration edges between artists

