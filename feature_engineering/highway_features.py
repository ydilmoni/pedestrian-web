#!/usr/bin/env python3
"""
highway_features.py

Modular highway type extraction for street networks.
Extracts and normalizes OSM highway tags with fallback strategies and validation.
Follows CLAUDE.md guidelines for production-ready, type-safe code.
"""
import os
import logging
from typing import Optional, Dict, Any, Union, List
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Configuration
class HighwayConfig:
    """Configuration constants for highway feature extraction."""
    CRS_METRIC = 3857  # EPSG:3857 for spatial operations
    BUFFER_DISTANCE = 10  # meters for fallback spatial matching
    DEFAULT_HIGHWAY_TYPE = 'unclassified'
    
    # Standard OSM highway types in order of preference
    HIGHWAY_HIERARCHY = [
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
        'residential', 'service', 'pedestrian', 'footway', 'path'
    ]
    
    @staticmethod
    def get_highway_priority(highway_type: str) -> int:
        """Get priority order for highway type (lower = higher priority).
        
        Args:
            highway_type: OSM highway type
            
        Returns:
            int: Priority order (0 = highest priority)
        """
        try:
            return HighwayConfig.HIGHWAY_HIERARCHY.index(highway_type)
        except ValueError:
            return len(HighwayConfig.HIGHWAY_HIERARCHY)  # Unknown types get lowest priority


class HighwayError(Exception):
    """Exception for highway processing errors."""
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


def normalize_highway_tag(highway_value: Union[str, List[str], None]) -> str:
    """Normalize highway tag value with intelligent prioritization.
    
    Handles various OSM highway tag formats:
    - Lists: selects highest priority highway type
    - Semicolon-separated strings: selects first valid type
    - Single values: validates and returns
    
    Args:
        highway_value: Raw highway tag value
        
    Returns:
        str: Normalized highway type
    """
    if highway_value is None:
        return HighwayConfig.DEFAULT_HIGHWAY_TYPE
    
    # Handle list of highway types - select by priority
    if isinstance(highway_value, list):
        if not highway_value:
            return HighwayConfig.DEFAULT_HIGHWAY_TYPE
        
        # Find highest priority highway type
        valid_highways = [h for h in highway_value if h and isinstance(h, str)]
        if not valid_highways:
            return HighwayConfig.DEFAULT_HIGHWAY_TYPE
        
        return min(valid_highways, key=HighwayConfig.get_highway_priority)
    
    # Handle string values
    if isinstance(highway_value, str):
        highway_value = highway_value.strip()
        if not highway_value:
            return HighwayConfig.DEFAULT_HIGHWAY_TYPE
        
        # Handle semicolon-separated values
        if ";" in highway_value:
            parts = [part.strip() for part in highway_value.split(";") if part.strip()]
            if not parts:
                return HighwayConfig.DEFAULT_HIGHWAY_TYPE
            return min(parts, key=HighwayConfig.get_highway_priority)
        
        return highway_value
    
    return HighwayConfig.DEFAULT_HIGHWAY_TYPE


def validate_highway_input(edges_gdf: gpd.GeoDataFrame) -> None:
    """Validate input GeoDataFrame for highway processing.
    
    Args:
        edges_gdf: Edge GeoDataFrame
        
    Raises:
        HighwayError: If input is invalid
    """
    if not isinstance(edges_gdf, gpd.GeoDataFrame):
        raise HighwayError("Input must be a GeoDataFrame", code=400)
    
    if edges_gdf.empty:
        raise HighwayError("GeoDataFrame cannot be empty", code=400)
    
    if 'osmid' not in edges_gdf.columns:
        raise HighwayError("GeoDataFrame must contain 'osmid' column", code=400)


def compute_highway(edges_gdf: gpd.GeoDataFrame, 
                   sensor_lookup: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
    """Attach highway column to edges GeoDataFrame with intelligent tag processing.
    
    This function processes OSM highway tags with multiple fallback strategies:
    1. Direct tag lookup from edges' existing highway column
    2. Spatial fallback using sensor lookup if provided
    3. Default assignment to 'unclassified' type
    
    Args:
        edges_gdf: Edge GeoDataFrame with 'osmid' and optionally 'highway' columns
        sensor_lookup: Optional sensor GeoDataFrame for spatial fallback matching
        
    Returns:
        GeoDataFrame: Copy of edges_gdf with normalized 'highway' column
        
    Raises:
        HighwayError: If input validation fails
    """
    # Input validation
    validate_highway_input(edges_gdf)
    
    try:
        # Process highway tags with normalization
        result_gdf = _process_highway_tags(edges_gdf)
        
        # Apply spatial fallback if sensor lookup provided
        if sensor_lookup is not None:
            result_gdf = _apply_spatial_fallback(result_gdf, sensor_lookup)
        
        # Final cleanup and validation
        result_gdf = _finalize_highway_column(result_gdf)
        
        logging.info(f"Successfully processed highway tags for {len(result_gdf)} edges")
        return result_gdf
        
    except Exception as e:
        raise HighwayError(
            f"Failed to compute highway features: {str(e)}",
            code=500,
            details={"n_edges": len(edges_gdf)}
        )


def _normalize_osmid_for_lookup(osmid_value: Union[str, int, List[Union[str, int]], None]) -> Optional[str]:
    """Normalize OSMID value for lookup operations.
    
    Args:
        osmid_value: Raw OSMID value
        
    Returns:
        str or None: Normalized OSMID string
    """
    if osmid_value is None:
        return None
    
    if isinstance(osmid_value, list):
        return str(osmid_value[0]) if osmid_value else None
    
    return str(osmid_value)


def _process_highway_tags(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Process and normalize highway tags from edges.
    
    Args:
        edges_gdf: Input edges GeoDataFrame
        
    Returns:
        GeoDataFrame: Edges with processed highway tags
    """
    result_gdf = edges_gdf.copy()
    
    # Normalize OSMIDs for consistent processing
    result_gdf['osmid_normalized'] = result_gdf['osmid'].apply(_normalize_osmid_for_lookup)
    
    # Process highway tags if column exists
    if 'highway' in result_gdf.columns:
        result_gdf['highway'] = result_gdf['highway'].apply(normalize_highway_tag)
    else:
        result_gdf['highway'] = HighwayConfig.DEFAULT_HIGHWAY_TYPE
    
    return result_gdf


def _apply_spatial_fallback(edges_gdf: gpd.GeoDataFrame, 
                           sensor_lookup: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Apply spatial fallback for missing highway tags using sensor data.
    
    Args:
        edges_gdf: Edges with potentially missing highway tags
        sensor_lookup: Sensor GeoDataFrame for spatial matching
        
    Returns:
        GeoDataFrame: Edges with spatial fallback applied
    """
    # Identify edges with missing highway information
    missing_mask = (edges_gdf['highway'] == HighwayConfig.DEFAULT_HIGHWAY_TYPE) | edges_gdf['highway'].isna()
    
    if not missing_mask.any() or 'geometry' not in edges_gdf.columns:
        return edges_gdf
    
    logging.info(f"Applying spatial fallback for {missing_mask.sum()} edges")
    
    # Prepare sensor lookup data
    sensor_buf = sensor_lookup.copy()
    if sensor_buf.crs != f"EPSG:{HighwayConfig.CRS_METRIC}":
        sensor_buf = sensor_buf.to_crs(epsg=HighwayConfig.CRS_METRIC)
    
    # Convert edges to metric CRS for accurate buffering
    edges_metric = edges_gdf.to_crs(epsg=HighwayConfig.CRS_METRIC)
    
    # Build spatial index for efficient querying
    if hasattr(sensor_buf, 'sindex'):
        sensor_sindex = sensor_buf.sindex
    else:
        logging.warning("Spatial index not available, using slower spatial operations")
        sensor_sindex = None
    
    # Apply spatial fallback to missing entries
    for idx in edges_gdf[missing_mask].index:
        if idx in edges_metric.index and 'geometry' in edges_metric.columns:
            geom = edges_metric.at[idx, 'geometry']
            if geom is not None:
                fallback_highway = _find_nearest_highway_tag(geom, sensor_buf, sensor_sindex)
                if fallback_highway:
                    edges_gdf.at[idx, 'highway'] = fallback_highway
    
    return edges_gdf


def _find_nearest_highway_tag(edge_geometry, sensor_gdf: gpd.GeoDataFrame, 
                              spatial_index) -> Optional[str]:
    """Find nearest highway tag using spatial proximity to sensors.
    
    Args:
        edge_geometry: Edge geometry
        sensor_gdf: Sensor GeoDataFrame with highway information
        spatial_index: Spatial index for efficient querying
        
    Returns:
        str or None: Highway tag from nearest sensor
    """
    try:
        # Buffer the edge geometry
        buffered = edge_geometry.buffer(HighwayConfig.BUFFER_DISTANCE)
        
        # Find nearby sensors
        if spatial_index is not None:
            candidate_indices = list(spatial_index.intersection(buffered.bounds))
            candidates = sensor_gdf.iloc[candidate_indices]
        else:
            # Fallback to full spatial query
            candidates = sensor_gdf[sensor_gdf.geometry.intersects(buffered)]
        
        if candidates.empty:
            return None
        
        # Extract highway tags from candidates
        highway_tags = candidates['highway'].dropna()
        if highway_tags.empty:
            return None
        
        # Return most common highway type (mode)
        return highway_tags.mode().iloc[0] if len(highway_tags.mode()) > 0 else None
        
    except Exception as e:
        logging.warning(f"Spatial fallback failed: {e}")
        return None


def _finalize_highway_column(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Finalize highway column with cleanup and validation.
    
    Args:
        edges_gdf: Edges with processed highway tags
        
    Returns:
        GeoDataFrame: Final edges with clean highway column
    """
    result_gdf = edges_gdf.copy()
    
    # Remove temporary columns
    temp_columns = [col for col in result_gdf.columns if col.endswith('_normalized') or col.endswith('_direct')]
    if temp_columns:
        result_gdf = result_gdf.drop(columns=temp_columns)
    
    # Final validation and cleanup
    result_gdf['highway'] = result_gdf['highway'].fillna(HighwayConfig.DEFAULT_HIGHWAY_TYPE)
    result_gdf['highway'] = result_gdf['highway'].apply(normalize_highway_tag)
    
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
    """Example of how to use the modular highway functions."""
    try:
        # Create test edges GeoDataFrame
        test_edges = gpd.GeoDataFrame({
            'osmid': ['123', ['456', '789'], '101112'],
            'highway': ['primary', ['secondary', 'tertiary'], 'residential;service'],
            'geometry': [None] * 3  # Minimal example
        })
        
        # Process highway tags
        result = compute_highway(test_edges)
        print(f"Processed highway tags for {len(result)} edges")
        print(f"Highway types: {result['highway'].unique()}")
        
        # Test normalization function
        test_values = ['primary', ['secondary', 'tertiary'], 'residential;service', None]
        for val in test_values:
            normalized = normalize_highway_tag(val)
            print(f"'{val}' -> '{normalized}'")
        
    except Exception as e:
        print(f"Example failed: {e}")


def main():
    """CLI entrypoint (DEPRECATED - use API instead).
    
    This function is kept for backward compatibility but the dynamic API
    approach is recommended for new implementations.
    """
    logging.warning("CLI main() function is deprecated. Use the dynamic API approach instead.")
    print("This CLI function has been deprecated. Please use the Flask API with dynamic highway processing.")
    print("Example: Use compute_highway(edges_gdf) in your API workflow")
    return


if __name__ == "__main__":
    example_usage()