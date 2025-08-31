#!/usr/bin/env python3
"""
centrality_features.py

Modular centrality feature extraction for street networks.
Computes betweenness and closeness centrality with performance optimizations and validation.
Follows CLAUDE.md guidelines for production-ready, type-safe code.
"""
import sys
import ast
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox

# Configuration
class CentralityConfig:
    """Configuration constants for centrality computation."""
    DEFAULT_SAMPLE_SIZE = 500
    CRS_METRIC = 3857  # EPSG:3857 for accurate distance calculations
    MIN_NODES_FOR_SAMPLING = 50
    CENTRALITY_COLUMNS = ['betweenness', 'closeness']
    
    @staticmethod
    def get_optimal_sample_size(n_nodes: int, requested_size: Optional[int] = None) -> Optional[int]:
        """Determine optimal sample size for centrality computation.
        
        Args:
            n_nodes: Total number of nodes in graph
            requested_size: User-requested sample size
            
        Returns:
            int or None: Optimal sample size, None for exact computation
        """
        if n_nodes < CentralityConfig.MIN_NODES_FOR_SAMPLING:
            return None  # Use exact computation for small graphs
        
        if requested_size is not None:
            return min(requested_size, n_nodes)
        
        return min(CentralityConfig.DEFAULT_SAMPLE_SIZE, n_nodes)


class CentralityError(Exception):
    """Exception for centrality computation errors."""
    def __init__(self, message: str, code: int = 400, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON responses."""
        return {
            "error": self.message,
            "code": self.code,
            "details": self.details
        }


def normalize_osm_id(osmid_value: Union[int, str, List[int], List[str]]) -> List[int]:
    """Normalize an OSMID field into a list of integers with comprehensive validation.
    
    Args:
        osmid_value: Raw OSMID value in various formats
        
    Returns:
        List[int]: Normalized list of OSMID integers
        
    Raises:
        CentralityError: If OSMID cannot be normalized
    """
    try:
        if isinstance(osmid_value, list):
            return [int(i) for i in osmid_value if i is not None]
        
        if isinstance(osmid_value, str):
            s = osmid_value.strip()
            if not s:
                return []
                
            # Handle string representation of lists: "[123, 456]"
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    return [int(i) for i in parsed if i is not None]
                except (ValueError, SyntaxError):
                    # Fallback: manual parsing "[123, 456]"
                    content = s[1:-1].strip()
                    if not content:
                        return []
                    parts = [part.strip() for part in content.split(",")]
                    return [int(part) for part in parts if part.isdigit()]
            
            # Handle single numeric string
            elif s.isdigit():
                return [int(s)]
        
        # Handle single numeric value
        if osmid_value is not None:
            return [int(osmid_value)]
            
        return []
        
    except (ValueError, TypeError) as e:
        raise CentralityError(
            f"Cannot normalize OSMID value: {osmid_value}",
            code=400,
            details={"osmid_value": str(osmid_value), "error": str(e)}
        )


def validate_graph_input(G: Union[nx.Graph, nx.DiGraph]) -> None:
    """Validate graph input for centrality computation.
    
    Args:
        G: NetworkX graph (directed or undirected)
        
    Raises:
        CentralityError: If graph is invalid
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise CentralityError("Input must be a NetworkX Graph", code=400)
    
    if len(G.nodes) == 0:
        raise CentralityError("Graph cannot be empty", code=400)
    
    # Check connectivity based on graph type
    if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
        if not nx.is_weakly_connected(G):
            logging.warning("Directed graph is not weakly connected. Using largest component.")
    else:
        if not nx.is_connected(G):
            logging.warning("Graph is not connected. Using largest connected component.")


def validate_edges_input(edges_gdf: gpd.GeoDataFrame) -> None:
    """Validate edges GeoDataFrame for centrality computation.
    
    Args:
        edges_gdf: Edge GeoDataFrame
        
    Raises:
        CentralityError: If edges are invalid
    """
    if not isinstance(edges_gdf, gpd.GeoDataFrame):
        raise CentralityError("edges_gdf must be a GeoDataFrame", code=400)
    
    if edges_gdf.empty:
        raise CentralityError("edges_gdf cannot be empty", code=400)
    
    if 'u' not in edges_gdf.columns:
        raise CentralityError("edges_gdf must contain 'u' column for source node IDs", code=400)


def compute_centrality(G: Union[nx.Graph, nx.DiGraph], 
                      edges_gdf: gpd.GeoDataFrame, 
                      sample_size: Optional[int] = None) -> gpd.GeoDataFrame:
    """Compute and attach betweenness & closeness centrality to edge GeoDataFrame.
    
    This function computes node centrality measures and maps them to edges based on
    source node IDs. Uses sampling for large graphs to maintain performance.
    
    Args:
        G: NetworkX graph with 'length' edge attributes
        edges_gdf: Edge GeoDataFrame with 'u' column for source node IDs
        sample_size: Number of nodes for approximate betweenness. Auto-determined if None.
        
    Returns:
        GeoDataFrame: Copy of edges_gdf with added 'betweenness' and 'closeness' columns
        
    Raises:
        CentralityError: If inputs are invalid or computation fails
    """
    # Input validation
    validate_graph_input(G)
    validate_edges_input(edges_gdf)
    
    # Convert directed graph to undirected for centrality computation
    if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
        G = G.to_undirected()
        logging.info("Converted directed graph to undirected for centrality computation")
    
    # Handle disconnected graphs by using largest component
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        logging.info(f"Using largest connected component: {len(G.nodes)} nodes")
    
    # Determine optimal sample size
    n_nodes = len(G.nodes)
    k = CentralityConfig.get_optimal_sample_size(n_nodes, sample_size)
    
    logging.info(f"Computing centrality for {n_nodes} nodes" + 
                (f" (sampling {k})" if k else " (exact)"))
    
    try:
        # Compute centrality measures
        centrality_data = _compute_centrality_measures(G, k)
        
        # Map to edges
        result_gdf = _map_centrality_to_edges(edges_gdf, centrality_data)
        
        logging.info(f"Successfully computed centrality for {len(result_gdf)} edges")
        return result_gdf
        
    except Exception as e:
        raise CentralityError(
            f"Failed to compute centrality: {str(e)}",
            code=500,
            details={"n_nodes": n_nodes, "sample_size": k}
        )


def _compute_centrality_measures(G: nx.Graph, sample_size: Optional[int]) -> Dict[str, Dict[int, float]]:
    """Compute betweenness and closeness centrality measures.
    
    Args:
        G: NetworkX graph
        sample_size: Sample size for betweenness computation
        
    Returns:
        dict: Centrality measures by node ID
    """
    # Compute betweenness centrality (potentially sampled)
    betweenness = nx.betweenness_centrality(
        G, 
        k=sample_size, 
        normalized=True, 
        weight="length"
    )
    
    # Compute closeness centrality (always exact)
    closeness = nx.closeness_centrality(G, distance="length")
    
    return {
        'betweenness': betweenness,
        'closeness': closeness
    }


def _map_centrality_to_edges(edges_gdf: gpd.GeoDataFrame, 
                            centrality_data: Dict[str, Dict[int, float]]) -> gpd.GeoDataFrame:
    """Map node centrality values to edges based on source node.
    
    Args:
        edges_gdf: Edge GeoDataFrame with 'u' column
        centrality_data: Centrality measures by node ID
        
    Returns:
        GeoDataFrame: Edges with centrality columns added
    """
    result_gdf = edges_gdf.copy()
    
    # Map centrality values to edges via source node 'u'
    for measure_name, node_values in centrality_data.items():
        result_gdf[measure_name] = result_gdf['u'].map(node_values).fillna(0.0)
    
    return result_gdf


def create_error_response(message: str, code: int = 400, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create consistent error response following CLAUDE.md format.
    
    Args:
        message: Error message
        code: HTTP status code
        details: Additional error details
        
    Returns:
        dict: Structured error response
    """
    return {
        "error": message,
        "code": code,
        "details": details
    }


def example_usage():
    """Example of how to use the modular centrality functions."""
    # Example: Create a simple test graph
    try:
        import networkx as nx
        
        # Create test graph
        G = nx.path_graph(5)
        nx.set_edge_attributes(G, 1.0, 'length')  # Add length attribute
        
        # Create test edges GeoDataFrame
        test_edges = gpd.GeoDataFrame({
            'u': [0, 1, 2, 3],
            'v': [1, 2, 3, 4],
            'geometry': [None] * 4  # Minimal example
        })
        
        # Compute centrality
        result = compute_centrality(G, test_edges)
        print(f"Computed centrality for {len(result)} edges")
        print(f"Centrality columns: {[col for col in result.columns if col in CentralityConfig.CENTRALITY_COLUMNS]}")
        
    except Exception as e:
        print(f"Example failed: {e}")


def main():
    """CLI entrypoint (DEPRECATED - use API instead).
    
    This function is kept for backward compatibility but the dynamic API
    approach is recommended for new implementations.
    """
    logging.warning("CLI main() function is deprecated. Use the dynamic API approach instead.")
    print("This CLI function has been deprecated. Please use the Flask API with dynamic centrality computation.")
    print("Example: Use compute_centrality(G, edges_gdf) in your API workflow")
    return

if __name__ == "__main__":
    example_usage()
