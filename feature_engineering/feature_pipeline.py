#!/usr/bin/env python3
"""
feature_pipeline.py

Unified feature extraction pipeline for pedestrian volume prediction.
Combines land use, centrality, highway, and temporal features into a single workflow.
Follows CLAUDE.md guidelines for production-ready, modular, and type-safe code.
"""
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, Union, List
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx

from .landuse_features import (
    get_landuse_polygons, 
    compute_landuse_edges,
    LandUseError
)
from .centrality_features import (
    compute_centrality,
    CentralityError
)
from .highway_features import (
    compute_highway,
    HighwayError
)
from .time_features import compute_time_features

# Configuration
class PipelineConfig:
    """Configuration constants for the feature extraction pipeline."""
    
    # Model feature columns in expected order
    FEATURE_COLUMNS = [
        "length",           # Edge length in meters
        "betweenness",      # Betweenness centrality
        "closeness",        # Closeness centrality  
        "Hour",            # Hour of day (0-23)
        "is_weekend",      # Boolean weekend flag
        "time_of_day",     # Categorical: morning/afternoon/evening/night
        "land_use",        # Categorical: residential/retail/commercial/other
        "highway",         # Categorical: primary/secondary/residential/etc
    ]
    
    # Categorical features for model processing
    CATEGORICAL_COLUMNS = ["time_of_day", "land_use", "highway"]
    
    # Network processing settings
    NETWORK_TYPE = "walk"
    CRS_METRIC = 3857  # EPSG:3857 for accurate length calculations
    
    # Performance settings
    MAX_NODES_FOR_EXACT_CENTRALITY = 1000
    CENTRALITY_SAMPLE_SIZE = 500

class PipelineError(Exception):
    """Exception for feature pipeline errors."""
    def __init__(self, message: str, code: int = 500, details: Optional[Dict[str, Any]] = None):
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

def validate_pipeline_inputs(place: Optional[str] = None, 
                           bbox: Optional[Tuple[float, float, float, float]] = None,
                           timestamp: Optional[Union[str, datetime]] = None) -> None:
    """Validate pipeline input parameters.
    
    Args:
        place: Place name for OSM query
        bbox: Bounding box coordinates  
        timestamp: Timestamp for temporal features
        
    Raises:
        PipelineError: If inputs are invalid
    """
    if place is None and bbox is None:
        raise PipelineError(
            "Either 'place' or 'bbox' must be provided",
            code=400,
            details={"place": place, "bbox": bbox}
        )
    
    if place and not isinstance(place, str):
        raise PipelineError("Place must be a string", code=400)
    
    if bbox and (not isinstance(bbox, (list, tuple)) or len(bbox) != 4):
        raise PipelineError("Bbox must be a tuple/list of 4 coordinates", code=400)

def extract_street_network(place: Optional[str] = None, 
                          bbox: Optional[Tuple[float, float, float, float]] = None) -> Tuple[nx.Graph, gpd.GeoDataFrame]:
    """Extract street network graph and edges for the specified location.
    
    Args:
        place: Place name (e.g., "Monaco", "Tel Aviv")
        bbox: Bounding box as (west, south, east, north) in EPSG:4326
        
    Returns:
        tuple: (NetworkX graph, edges GeoDataFrame)
        
    Raises:
        PipelineError: If network extraction fails
    """
    try:
        logging.info(f"Extracting street network for {place or 'bbox'}")
        
        # Download street network
        if place:
            G = ox.graph_from_place(place, network_type=PipelineConfig.NETWORK_TYPE)
        else:
            # OSMnx 2.0+ expects bbox as (west, south, east, north) tuple
            bbox_osmnx = (bbox[0], bbox[1], bbox[2], bbox[3])  # (west, south, east, north)
            G = ox.graph_from_bbox(
                bbox=bbox_osmnx,
                network_type=PipelineConfig.NETWORK_TYPE
            )
        
        # Project to metric CRS for accurate length calculation
        G_proj = ox.project_graph(G, to_crs=f"EPSG:{PipelineConfig.CRS_METRIC}")
        
        # Convert to GeoDataFrame
        edges_gdf = ox.graph_to_gdfs(G_proj, nodes=False, edges=True)
        edges_gdf = edges_gdf.reset_index()
        
        # Add length column in meters
        edges_gdf['length'] = edges_gdf.geometry.length
        
        # Convert back to EPSG:4326 for consistency with other modules
        edges_gdf = edges_gdf.to_crs("EPSG:4326")
        G = ox.project_graph(G_proj, to_crs="EPSG:4326")
        
        logging.info(f"Extracted network: {len(G.nodes)} nodes, {len(edges_gdf)} edges")
        return G, edges_gdf
        
    except Exception as e:
        raise PipelineError(
            f"Failed to extract street network: {str(e)}",
            code=500,
            details={"place": place, "bbox": bbox}
        )

def extract_all_features(edges_gdf: gpd.GeoDataFrame,
                        graph: nx.Graph,
                        place: Optional[str] = None,
                        bbox: Optional[Tuple[float, float, float, float]] = None,
                        timestamp: Optional[Union[str, datetime]] = None) -> gpd.GeoDataFrame:
    """Extract all features for the edges using the modular feature functions.
    
    Args:
        edges_gdf: Street edges GeoDataFrame
        graph: NetworkX graph for centrality computation
        place: Place name for land use data
        bbox: Bounding box for land use data
        timestamp: Timestamp for temporal features
        
    Returns:
        GeoDataFrame: Edges with all features extracted
        
    Raises:
        PipelineError: If feature extraction fails
    """
    try:
        result_gdf = edges_gdf.copy()
        extraction_times = {}
        
        # 1. Extract land use features
        start_time = time.time()
        try:
            result_gdf = compute_landuse_edges(
                result_gdf, 
                place=place, 
                bbox=bbox
            )
            extraction_times['landuse'] = time.time() - start_time
            logging.info(f"Land use extraction completed in {extraction_times['landuse']:.2f}s")
        except LandUseError as e:
            logging.error(f"Land use extraction failed: {e.message}")
            # Graceful degradation: assign default land use
            result_gdf['land_use'] = 'other'
            extraction_times['landuse'] = time.time() - start_time
        
        # 2. Extract centrality features
        start_time = time.time()
        try:
            # Use sampling for large graphs
            sample_size = None
            if len(graph.nodes) > PipelineConfig.MAX_NODES_FOR_EXACT_CENTRALITY:
                sample_size = PipelineConfig.CENTRALITY_SAMPLE_SIZE
            
            result_gdf = compute_centrality(graph, result_gdf, sample_size=sample_size)
            extraction_times['centrality'] = time.time() - start_time
            logging.info(f"Centrality extraction completed in {extraction_times['centrality']:.2f}s")
        except CentralityError as e:
            logging.error(f"Centrality extraction failed: {e.message}")
            # Graceful degradation: assign default centrality
            result_gdf['betweenness'] = 0.0
            result_gdf['closeness'] = 0.0
            extraction_times['centrality'] = time.time() - start_time
        
        # 3. Extract highway features  
        start_time = time.time()
        try:
            result_gdf = compute_highway(result_gdf)
            extraction_times['highway'] = time.time() - start_time
            logging.info(f"Highway extraction completed in {extraction_times['highway']:.2f}s")
        except HighwayError as e:
            logging.error(f"Highway extraction failed: {e.message}")
            # Graceful degradation: assign default highway type
            result_gdf['highway'] = 'unclassified'
            extraction_times['highway'] = time.time() - start_time
        
        # 4. Extract temporal features
        start_time = time.time()
        result_gdf = compute_time_features(result_gdf, timestamp=timestamp)
        extraction_times['temporal'] = time.time() - start_time
        logging.info(f"Temporal extraction completed in {extraction_times['temporal']:.2f}s")
        
        # Log total extraction time
        total_time = sum(extraction_times.values())
        logging.info(f"Total feature extraction completed in {total_time:.2f}s")
        logging.info(f"Extraction breakdown: {extraction_times}")
        
        return result_gdf
        
    except Exception as e:
        raise PipelineError(
            f"Feature extraction failed: {str(e)}",
            code=500,
            details={"n_edges": len(edges_gdf)}
        )

def validate_features(features_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Validate extracted features and return validation summary.
    
    Args:
        features_gdf: GeoDataFrame with extracted features
        
    Returns:
        dict: Validation summary with statistics
        
    Raises:
        PipelineError: If critical validation fails
    """
    validation_summary = {
        "n_edges": len(features_gdf),
        "missing_features": {},
        "feature_stats": {},
        "warnings": []
    }
    
    # Check for missing required columns
    missing_columns = [col for col in PipelineConfig.FEATURE_COLUMNS if col not in features_gdf.columns]
    if missing_columns:
        raise PipelineError(
            f"Missing required feature columns: {missing_columns}",
            code=500,
            details={"missing_columns": missing_columns}
        )
    
    # Validate each feature
    for col in PipelineConfig.FEATURE_COLUMNS:
        if col in features_gdf.columns:
            series = features_gdf[col]
            
            # Count missing values
            n_missing = series.isna().sum()
            validation_summary["missing_features"][col] = n_missing
            
            if n_missing > 0:
                validation_summary["warnings"].append(f"{col}: {n_missing} missing values")
            
            # Collect statistics
            if col in ["length", "betweenness", "closeness", "Hour"]:
                validation_summary["feature_stats"][col] = {
                    "min": float(series.min()) if not series.empty else None,
                    "max": float(series.max()) if not series.empty else None,
                    "mean": float(series.mean()) if not series.empty else None
                }
            elif col in PipelineConfig.CATEGORICAL_COLUMNS:
                validation_summary["feature_stats"][col] = {
                    "unique_values": series.value_counts().to_dict()
                }
    
    # Validate data ranges
    if 'Hour' in features_gdf.columns:
        hour_range = features_gdf['Hour'].dropna()
        if not hour_range.empty and (hour_range < 0).any() or (hour_range > 23).any():
            validation_summary["warnings"].append("Hour values outside valid range [0-23]")
    
    if 'length' in features_gdf.columns:
        length_values = features_gdf['length'].dropna()
        if not length_values.empty and (length_values <= 0).any():
            validation_summary["warnings"].append("Non-positive length values found")
    
    logging.info(f"Feature validation completed: {len(validation_summary['warnings'])} warnings")
    
    return validation_summary

def run_feature_pipeline(place: Optional[str] = None,
                        bbox: Optional[Tuple[float, float, float, float]] = None,
                        timestamp: Optional[Union[str, datetime]] = None) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Run the complete feature extraction pipeline.
    
    This is the main entry point that orchestrates all feature extraction steps:
    1. Input validation
    2. Street network extraction
    3. Feature extraction (land use, centrality, highway, temporal)
    4. Feature validation
    5. Return processed data ready for model prediction
    
    Args:
        place: Place name (e.g., "Monaco", "Tel Aviv") 
        bbox: Bounding box as (minx, miny, maxx, maxy) in EPSG:4326
        timestamp: Timestamp for temporal features (ISO format or datetime)
        
    Returns:
        tuple: (features_gdf, pipeline_metadata)
            - features_gdf: GeoDataFrame with all features extracted
            - pipeline_metadata: Dict with processing statistics and validation info
            
    Raises:
        PipelineError: If any step in the pipeline fails
    """
    pipeline_start = time.time()
    
    try:
        # 1. Validate inputs
        validate_pipeline_inputs(place, bbox, timestamp)
        logging.info(f"Starting feature pipeline for {place or 'bbox'}")
        
        # 2. Extract street network
        graph, edges_gdf = extract_street_network(place, bbox)
        
        # 3. Extract all features
        features_gdf = extract_all_features(
            edges_gdf, graph, place=place, bbox=bbox, timestamp=timestamp
        )
        
        # 4. Validate results
        validation_summary = validate_features(features_gdf)
        
        # 5. Compile metadata
        pipeline_metadata = {
            "processing_time": time.time() - pipeline_start,
            "location": {"place": place, "bbox": bbox},
            "timestamp": str(timestamp) if timestamp else None,
            "network_stats": {
                "n_nodes": len(graph.nodes),
                "n_edges": len(features_gdf)
            },
            "validation": validation_summary,
            "feature_columns": PipelineConfig.FEATURE_COLUMNS,
            "categorical_columns": PipelineConfig.CATEGORICAL_COLUMNS
        }
        
        logging.info(f"Pipeline completed successfully in {pipeline_metadata['processing_time']:.2f}s")
        
        return features_gdf, pipeline_metadata
        
    except Exception as e:
        if isinstance(e, PipelineError):
            raise
        else:
            raise PipelineError(
                f"Pipeline execution failed: {str(e)}",
                code=500,
                details={"place": place, "bbox": bbox}
            )

def prepare_model_features(features_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Prepare features for model prediction by selecting and ordering columns.
    
    Args:
        features_gdf: GeoDataFrame with extracted features
        
    Returns:
        DataFrame: Features ready for model input with correct column order
        
    Raises:
        PipelineError: If required features are missing
    """
    try:
        # Select only the required feature columns in correct order
        model_features = features_gdf[PipelineConfig.FEATURE_COLUMNS].copy()
        
        # Validate no missing values in critical features
        critical_nulls = model_features.isnull().sum()
        if critical_nulls.any():
            logging.warning(f"Found null values in features: {critical_nulls[critical_nulls > 0].to_dict()}")
        
        # Fill any remaining nulls with defaults
        defaults = {
            'length': 0.0,
            'betweenness': 0.0,
            'closeness': 0.0,
            'Hour': 12,  # Default to noon
            'is_weekend': 0,
            'time_of_day': 'afternoon',
            'land_use': 'other',
            'highway': 'unclassified'
        }
        
        for col, default_val in defaults.items():
            if col in model_features.columns:
                model_features[col] = model_features[col].fillna(default_val)
        
        logging.info(f"Prepared {len(model_features)} feature vectors for model input")
        
        return model_features
        
    except Exception as e:
        raise PipelineError(
            f"Failed to prepare model features: {str(e)}",
            code=500,
            details={"available_columns": list(features_gdf.columns)}
        )

def example_usage():
    """Example of how to use the feature pipeline."""
    try:
        # Run pipeline for Monaco
        features_gdf, metadata = run_feature_pipeline(
            place="Monaco",
            timestamp="2024-01-15T14:30:00"
        )
        
        print(f"Pipeline completed for {metadata['network_stats']['n_edges']} edges")
        print(f"Processing time: {metadata['processing_time']:.2f}s")
        print(f"Features: {list(features_gdf.columns)}")
        
        # Prepare for model
        model_features = prepare_model_features(features_gdf)
        print(f"Model features ready: {model_features.shape}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    example_usage()