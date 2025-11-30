"""
Spotify Artist Network - Python Shiny App
==========================================

This app visualizes the Spotify artist collaboration network with interactive filters
and multiple visualizations including network graphs and scatter plots.
"""

import pandas as pd
import numpy as np
import networkx as nx
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("Loading data...")

# Load edges and nodes
edges_df = pd.read_csv('edges.csv')
nodes_df = pd.read_csv('nodes.csv')

# Handle duplicates in nodes: keep first occurrence
nodes_df = nodes_df.drop_duplicates(subset=['spotify_id'], keep='first')

# Handle missing values: fill missing followers with 0, drop rows with missing name
nodes_df['followers'] = nodes_df['followers'].fillna(0)
nodes_df = nodes_df.dropna(subset=['name'])

# Filter edges to only include nodes that exist in nodes.csv
# This ensures all nodes in the graph have metadata
valid_node_ids = set(nodes_df['spotify_id'].unique())
edges_df = edges_df[
    (edges_df['id_0'].isin(valid_node_ids)) & 
    (edges_df['id_1'].isin(valid_node_ids))
].copy()

print(f"Loaded {len(edges_df)} edges and {len(nodes_df)} nodes")

# Build NetworkX graph (undirected)
print("Building network graph...")
G = nx.Graph()
G.add_edges_from(zip(edges_df['id_0'], edges_df['id_1']))

# Compute degree for each node
degree_dict = dict(G.degree())

# Create a merged dataframe with all node attributes
# This includes: spotify_id, name, followers, popularity, degree
nodes_with_degree = nodes_df.copy()
nodes_with_degree['degree'] = nodes_with_degree['spotify_id'].map(degree_dict).fillna(0).astype(int)

# Only keep nodes that are in the graph (have at least one edge)
nodes_with_degree = nodes_with_degree[nodes_with_degree['degree'] > 0].copy()

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# For large graphs, we'll sample a subset for visualization
# We'll keep top nodes by degree to maintain important connections
MAX_NODES_FOR_VIZ = 10000
if len(nodes_with_degree) > MAX_NODES_FOR_VIZ:
    print(f"Graph is large ({len(nodes_with_degree)} nodes). Sampling top {MAX_NODES_FOR_VIZ} by degree...")
    top_nodes = nodes_with_degree.nlargest(MAX_NODES_FOR_VIZ, 'degree')['spotify_id'].tolist()
    G_viz = G.subgraph(top_nodes).copy()
    nodes_viz = nodes_with_degree[nodes_with_degree['spotify_id'].isin(top_nodes)].copy()
else:
    G_viz = G.copy()
    nodes_viz = nodes_with_degree.copy()

# Compute centrality metrics on the visualization subgraph
print("Computing eigenvector centrality...")
try:
    eigenvector_centrality = nx.eigenvector_centrality(G_viz, max_iter=100)
except:
    # Fallback if eigenvector centrality doesn't converge
    print("Eigenvector centrality didn't converge, using degree centrality as fallback")
    eigenvector_centrality = nx.degree_centrality(G_viz)

print("Computing betweenness centrality (connectivity score)...")
# Use approximate betweenness for large graphs (faster)
if G_viz.number_of_nodes() > 5000:
    betweenness_centrality = nx.betweenness_centrality(G_viz, k=500)  # Sample 500 nodes
else:
    betweenness_centrality = nx.betweenness_centrality(G_viz)

# Add centrality metrics to nodes_viz
nodes_viz['eigenvector_centrality'] = nodes_viz['spotify_id'].map(eigenvector_centrality).fillna(0)
nodes_viz['betweenness_centrality'] = nodes_viz['spotify_id'].map(betweenness_centrality).fillna(0)

# Compute layout once (spring layout for 2D visualization)
# This is computationally expensive, so we do it once at startup
print("Computing graph layout (this may take a moment)...")
pos = nx.spring_layout(G_viz, k=1, iterations=50, seed=42)

# Prepare layout data for Plotly
node_x = [pos[node][0] for node in G_viz.nodes()]
node_y = [pos[node][1] for node in G_viz.nodes()]

# Create a mapping from spotify_id to index for edge drawing
node_to_idx = {node: idx for idx, node in enumerate(G_viz.nodes())}

print("Data preprocessing complete!")

# ============================================================================
# UI DEFINITION
# ============================================================================

app_ui = ui.page_sidebar(
    # Sidebar with filters
    ui.sidebar(
        ui.h2("Filters", class_="text-center"),
        ui.hr(),
        
        # Minimum degree slider
        ui.input_slider(
            "min_degree",
            "Minimum Degree",
            min=0,
            max=int(nodes_viz['degree'].max()),
            value=0,
            step=1
        ),
        ui.p("Filter nodes by number of collaborations (degree).", class_="text-muted small"),
        
        ui.hr(),
        
        # Minimum followers slider (shared across visualizations)
        # Range: 0 to 10 million, step by 100k for easier filtering
        ui.input_slider(
            "min_followers",
            "Minimum Followers",
            min=0,
            max=10_000_000,
            value=0,
            step=100_000
        ),
        ui.p("Filter artists by follower count. Affects all visualizations.", class_="text-muted small"),
        
        ui.hr(),
        
        # Minimum eigenvector centrality text input
        ui.input_text(
            "min_eigenvector",
            "Min Eigenvector Centrality",
            value="0",
            placeholder="e.g., 0.001"
        ),
        ui.p("Enter minimum eigenvector centrality (e.g., 0.001). Leave 0 for no filter.", class_="text-muted small"),
        
        ui.hr(),
        
        # Summary stats display
        ui.h4("Current Filter Summary"),
        ui.output_text_verbatim("filter_summary"),
        
        width=300
    ),
    
    # Main content area with labeled sections
    ui.div(
        # Collaboration Network section
        ui.h2("Collaboration Network", class_="mt-4 mb-3"),
        ui.p(
            "Interactive network graph showing artist collaborations. "
            "Hover over nodes to see artist details. Use the sidebar filters to focus on specific artists. "
            "Note: Degree values reflect ALL collaborations, but only edges between top 10,000 artists are shown.",
            class_="text-muted mb-3"
        ),
        ui.div(
            output_widget("network_graph"),
            style="width: 100%; min-height: 800px;"
        ),
        
        ui.hr(class_="my-5"),
        
        # Degree vs Popularity section
        ui.h2("Degree vs Popularity", class_="mt-4 mb-3"),
        ui.p(
            "Relationship between number of collaborations (degree) and Spotify popularity score.",
            class_="text-muted mb-3"
        ),
        ui.div(
            output_widget("degree_vs_popularity"),
            style="width: 100%; min-height: 500px;"
        ),
        
        ui.hr(class_="my-5"),
        
        # Popularity vs Followers section
        ui.h2("Popularity vs Followers", class_="mt-4 mb-3"),
        ui.p(
            "Relationship between Spotify popularity score and follower count.",
            class_="text-muted mb-3"
        ),
        ui.div(
            output_widget("popularity_vs_followers"),
            style="width: 100%; min-height: 500px;"
        ),
        
        ui.hr(class_="my-5"),
        
        # Degree Distribution section
        ui.h2("Degree Distribution", class_="mt-4 mb-3"),
        ui.p(
            "Histogram showing the distribution of node degrees (number of collaborations per artist).",
            class_="text-muted mb-3"
        ),
        ui.div(
            output_widget("degree_distribution"),
            style="width: 100%; min-height: 500px;"
        ),
        
        class_="p-4"
    ),
    title="Spotify Artist Network"
)

# ============================================================================
# SERVER LOGIC
# ============================================================================

def server(input, output, session):
    
    # Reactive data filtering based on slider inputs
    @reactive.calc
    def filtered_nodes():
        """
        Filter nodes based on minimum degree, minimum followers, and eigenvector centrality.
        This reactive calculation updates whenever the inputs change.
        """
        # Parse eigenvector centrality input (handle invalid input gracefully)
        try:
            min_eigen = float(input.min_eigenvector())
        except (ValueError, TypeError):
            min_eigen = 0.0
        
        filtered = nodes_viz[
            (nodes_viz['degree'] >= input.min_degree()) &
            (nodes_viz['followers'] >= input.min_followers()) &
            (nodes_viz['eigenvector_centrality'] >= min_eigen)
        ].copy()
        return filtered
    
    # Filter summary text
    @render.text
    def filter_summary():
        """Display summary statistics for the current filter settings."""
        filtered = filtered_nodes()
        total_nodes = len(nodes_viz)
        filtered_count = len(filtered)
        
        if filtered_count > 0:
            avg_degree = filtered['degree'].mean()
            avg_followers = filtered['followers'].mean()
            avg_popularity = filtered['popularity'].mean()
            
            return (
                f"Showing: {filtered_count:,} / {total_nodes:,} artists\n"
                f"Avg Degree: {avg_degree:.1f}\n"
                f"Avg Followers: {avg_followers:,.0f}\n"
                f"Avg Popularity: {avg_popularity:.1f}"
            )
        else:
            return "No artists match the current filters."
    
    # Network graph visualization
    @render_widget
    def network_graph():
        """
        Render the collaboration network graph with Plotly.
        Nodes are filtered by degree and followers, and edges are only shown
        between displayed nodes.
        """
        filtered = filtered_nodes()
        
        if len(filtered) == 0:
            # Return empty plot if no nodes match filters
            fig = go.Figure()
            fig.add_annotation(
                text="No nodes match the current filters. Adjust the sliders to see the network.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800,
                autosize=True
            )
            return fig
        
        # Get filtered node IDs
        filtered_ids = set(filtered['spotify_id'].tolist())
        
        # Create subgraph with only filtered nodes
        G_filtered = G_viz.subgraph(filtered_ids).copy()
        
        if G_filtered.number_of_nodes() == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No connected nodes match the filters.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800,
                autosize=True
            )
            return fig
        
        # Get positions for filtered nodes
        filtered_pos = {node: pos[node] for node in G_filtered.nodes() if node in pos}
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        for edge in G_filtered.edges():
            x0, y0 = filtered_pos[edge[0]]
            x1, y1 = filtered_pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node data
        node_info = filtered.set_index('spotify_id')
        node_x_filtered = []
        node_y_filtered = []
        node_text = []
        node_hover = []
        node_size = []
        node_colors = []
        
        for node_id in G_filtered.nodes():
            if node_id in filtered_pos:
                node_x_filtered.append(filtered_pos[node_id][0])
                node_y_filtered.append(filtered_pos[node_id][1])
                
                # Get node attributes
                if node_id in node_info.index:
                    row = node_info.loc[node_id]
                    name = row['name']
                    degree = row['degree']
                    followers = row['followers']
                    popularity = row['popularity']
                    eigenvector = row['eigenvector_centrality']
                    betweenness = row['betweenness_centrality']
                    
                    node_text.append(name)
                    node_hover.append(
                        f"<b>{name}</b><br>"
                        f"Degree: {degree}<br>"
                        f"Followers: {followers:,.0f}<br>"
                        f"Popularity: {popularity}<br>"
                        f"Eigenvector Centrality: {eigenvector:.4f}<br>"
                        f"Betweenness Centrality (Connectivity): {betweenness:.4f}"
                    )
                    # Size nodes by degree (with some scaling)
                    node_size.append(max(5, min(20, degree * 0.5 + 5)))
                    node_colors.append(popularity)
                else:
                    node_text.append(node_id)
                    node_hover.append(f"ID: {node_id}")
                    node_size.append(5)
                    node_colors.append(0)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x_filtered, y=node_y_filtered,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_hover,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=False,
                color=node_colors,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title=dict(text="Popularity"),
                    xanchor="left"
                ),
                line=dict(width=1, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=f"Collaboration Network ({G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges)",
                    x=0.5,
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Node size = degree, Color = popularity",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(size=12, color="#888")
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800,
                autosize=True
            )
        )
        
        # Ensure full width rendering
        fig.update_layout(
            template="plotly_white"
        )
        
        return fig
    
    # Degree vs Popularity scatter plot
    @render_widget
    def degree_vs_popularity():
        """
        Scatter plot showing the relationship between degree (collaborations)
        and popularity. Filtered by minimum followers.
        """
        filtered = filtered_nodes()
        
        if len(filtered) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No artists match the current filters.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                xaxis_title="Popularity",
                yaxis_title="Degree",
                height=500,
                autosize=True
            )
            return fig
        
        # Create hover text
        hover_text = [
            f"<b>{row['name']}</b><br>"
            f"Degree: {row['degree']}<br>"
            f"Popularity: {row['popularity']}<br>"
            f"Followers: {row['followers']:,.0f}<br>"
            f"Eigenvector Centrality: {row['eigenvector_centrality']:.4f}<br>"
            f"Betweenness Centrality (Connectivity): {row['betweenness_centrality']:.4f}"
            for _, row in filtered.iterrows()
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=filtered['popularity'],
            y=filtered['degree'],
            mode='markers',
            marker=dict(
                size=8,
                color=filtered['followers'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=dict(text="Followers")),
                line=dict(width=0.5, color='white'),
                opacity=0.7
            ),
            text=hover_text,
            hoverinfo='text',
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Degree vs Popularity",
            xaxis_title="Popularity",
            yaxis_title="Degree (Number of Collaborations)",
            height=500,
            hovermode='closest',
            autosize=True
        )
        
        return fig
    
    # Popularity vs Followers scatter plot
    @render_widget
    def popularity_vs_followers():
        """
        Scatter plot showing the relationship between popularity and followers.
        Filtered by minimum followers.
        """
        filtered = filtered_nodes()
        
        if len(filtered) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No artists match the current filters.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                xaxis_title="Popularity",
                yaxis_title="Followers",
                height=500,
                autosize=True
            )
            return fig
        
        # Create hover text
        hover_text = [
            f"<b>{row['name']}</b><br>"
            f"Popularity: {row['popularity']}<br>"
            f"Followers: {row['followers']:,.0f}<br>"
            f"Degree: {row['degree']}<br>"
            f"Eigenvector Centrality: {row['eigenvector_centrality']:.4f}<br>"
            f"Betweenness Centrality (Connectivity): {row['betweenness_centrality']:.4f}"
            for _, row in filtered.iterrows()
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=filtered['popularity'],
            y=filtered['followers'],
            mode='markers',
            marker=dict(
                size=8,
                color=filtered['degree'],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title=dict(text="Degree")),
                line=dict(width=0.5, color='white'),
                opacity=0.7
            ),
            text=hover_text,
            hoverinfo='text',
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Popularity vs Followers",
            xaxis_title="Popularity",
            yaxis_title="Followers",
            height=500,
            hovermode='closest',
            yaxis_type="log",  # Log scale for followers (wide range)
            autosize=True
        )
        
        return fig
    
    # Degree Distribution histogram
    @render_widget
    def degree_distribution():
        """
        Histogram showing the distribution of node degrees.
        Helps understand the network structure - most networks follow a power-law distribution.
        """
        filtered = filtered_nodes()
        
        if len(filtered) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No artists match the current filters.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                xaxis_title="Degree",
                yaxis_title="Count",
                height=500,
                autosize=True
            )
            return fig
        
        # Create histogram with bin size of 25 (0-24, 25-49, etc.)
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=filtered['degree'],
            xbins=dict(
                start=0,
                size=25  # Each bin spans 25 degrees (0-24, 25-49, etc.)
            ),
            marker=dict(
                color='#1DB954',  # Spotify green
                line=dict(width=1, color='white')
            ),
            hovertemplate='Degree Range: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Degree Distribution (Number of Collaborations per Artist)",
            xaxis_title="Degree (Number of Collaborations)",
            yaxis_title="Number of Artists",
            height=500,
            hovermode='closest',
            autosize=True,
            bargap=0.1
        )
        
        return fig

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = App(app_ui, server)
