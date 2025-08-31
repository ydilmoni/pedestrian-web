#!/usr/bin/env python3
"""
landuse_features.py

Modular land-use feature extraction supporting dynamic OSM data generation for any city.
Follows functional composition principles with comprehensive error handling and caching.
"""
import sys
import os
import hashlib
import time
from pathlib import Path
from typing import Optional, Set, Tuple, Dict, Any, Union
import pandas as pd
import geopandas as gpd
import fiona
import logging
import osmnx as ox
from shapely.geometry import Point

# Configuration
class LandUseConfig:
    """Configuration constants for land use processing."""
    BUFFER_METERS = 150
    CRS_METRIC = 3857  # EPSG:3857 for buffering
    DEFAULT_ALLOWED = {"residential", "retail", "commercial"}
    OSM_LANDUSE_TAGS = {"landuse": True}
    VALID_CATEGORIES = {"residential", "commercial", "retail", "industrial", 
                      "recreation_ground", "park"}
    CACHE_MAX_AGE_HOURS = 6
    
    @staticmethod
    def get_temp_dir() -> Path:
        """Get temp directory for caching, create if needed."""
        temp_dir = Path(__file__).parent.parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        return temp_dir


def validate_coordinates(bbox: Tuple[float, float, float, float]) -> bool:
    """Validate bounding box coordinates are within valid lat/lng ranges.
    
    Args:
        bbox: Bounding box as (minx, miny, maxx, maxy)
        
    Returns:
        bool: True if coordinates are valid
    """
    minx, miny, maxx, maxy = bbox
    return (-180 <= minx <= 180 and -180 <= maxx <= 180 and 
            -90 <= miny <= 90 and -90 <= maxy <= 90 and
            minx < maxx and miny < maxy)


def validate_place_name(place: str) -> bool:
    """Validate place name is non-empty and properly formatted.
    
    Args:
        place: Place name string
        
    Returns:
        bool: True if place name is valid
    """
    return bool(place and place.strip() and len(place.strip()) > 0)


def generate_cache_key(place: Optional[str] = None, 
                      bbox: Optional[Tuple[float, float, float, float]] = None) -> str:
    """Generate consistent cache key for land use data.
    
    Args:
        place: Place name
        bbox: Bounding box coordinates
        
    Returns:
        str: Hash-based cache key
    """
    if place:
        cache_input = place.strip().lower()
    elif bbox:
        cache_input = "_".join(map(str, bbox))
    else:
        raise ValueError("Either place or bbox must be provided")
    
    return hashlib.md5(cache_input.encode()).hexdigest()[:8]


def generate_cache_filename(place: Optional[str] = None, 
                          bbox: Optional[Tuple[float, float, float, float]] = None) -> str:
    """Generate cache filename for land use data.
    
    Args:
        place: Place name
        bbox: Bounding box coordinates
        
    Returns:
        str: Cache filename
    """
    cache_key = generate_cache_key(place, bbox)
    
    if place:
        safe_place = place.replace(' ', '_').replace(',', '_').replace('/', '_')
        return f"{safe_place}_{cache_key}_landuse.gpkg"
    else:
        return f"bbox_{cache_key}_landuse.gpkg"


def get_landuse_polygons(place: Optional[str] = None, 
                        bbox: Optional[Tuple[float, float, float, float]] = None, 
                        save_path: Optional[str] = None) -> gpd.GeoDataFrame:
    """Dynamically fetch and cache land use polygon layer for any place or bounding box.
    
    Args:
        place: Place name (e.g., "Monaco", "Melbourne, Australia")
        bbox: Bounding box as (minx, miny, maxx, maxy) in EPSG:4326
        save_path: Custom save path. If None, uses temp directory.
        
    Returns:
        GeoDataFrame: Land use polygons with standardized schema
        
    Raises:
        ValueError: If neither place nor bbox provided, or invalid coordinates
        OSError: If cache directory cannot be created
    """
    # Input validation
    if place is None and bbox is None:
        raise ValueError("Either 'place' or 'bbox' must be provided")
    
    if place and not validate_place_name(place):
        raise ValueError(f"Invalid place name: '{place}'")
    
    if bbox and not validate_coordinates(bbox):
        raise ValueError(f"Invalid bounding box coordinates: {bbox}")
    
    # Generate cache path
    if save_path is None:
        cache_name = generate_cache_filename(place, bbox)
        save_path = LandUseConfig.get_temp_dir() / cache_name
    
    # Check cache first
    if os.path.exists(save_path):
        logging.info(f"Loading cached land use data from {save_path}")
        try:
            return gpd.read_file(save_path)
        except Exception as e:
            logging.warning(f"Failed to load cached file {save_path}: {e}")
            # Continue to re-download
    
    # Download from OSM
    logging.info(f"Downloading land use data for {place or 'bbox'}")
    landuse = _fetch_osm_landuse(place, bbox)
    
    # Process and filter data
    landuse = _process_landuse_data(landuse)
    
    # Cache the result
    _save_landuse_cache(landuse, save_path)
    
    return landuse


def _fetch_osm_landuse(place: Optional[str], 
                      bbox: Optional[Tuple[float, float, float, float]]) -> gpd.GeoDataFrame:
    """Fetch land use data from OpenStreetMap.
    
    Args:
        place: Place name for OSM query
        bbox: Bounding box coordinates
        
    Returns:
        GeoDataFrame: Raw OSM land use data
        
    Raises:
        ValueError: If OSM query fails and no fallback available
    """
    tags = LandUseConfig.OSM_LANDUSE_TAGS
    
    if place:
        try:
            return ox.features_from_place(place, tags=tags)
        except Exception as e:
            logging.warning(f"Failed to get land use from place '{place}': {e}")
            if bbox is None:
                raise LandUseError(
                    f"Could not find place '{place}' and no bbox provided",
                    code=404,
                    details={"place": place, "osm_error": str(e)}
                )
            # Fallback to bbox
            # OSMnx 2.0+ expects bbox as (west, south, east, north) tuple
            bbox_osmnx = (bbox[0], bbox[1], bbox[2], bbox[3])  # (west, south, east, north)
            return ox.features_from_bbox(bbox=bbox_osmnx, tags=tags)
    else:
        # OSMnx 2.0+ expects bbox as (west, south, east, north) tuple  
        bbox_osmnx = (bbox[0], bbox[1], bbox[2], bbox[3])  # (west, south, east, north)
        return ox.features_from_bbox(bbox=bbox_osmnx, tags=tags)


def _process_landuse_data(landuse: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Process and filter raw OSM land use data.
    
    Args:
        landuse: Raw OSM land use GeoDataFrame
        
    Returns:
        GeoDataFrame: Processed land use data with standardized schema
    """
    if landuse.empty:
        logging.warning("No land use data found, creating empty GeoDataFrame")
        return gpd.GeoDataFrame(columns=['landuse', 'geometry'], crs="EPSG:4326")
    
    # Filter to valid categories
    if 'landuse' in landuse.columns:
        landuse = landuse[landuse["landuse"].isin(LandUseConfig.VALID_CATEGORIES)]
    
    # Ensure required columns exist
    if 'landuse' not in landuse.columns:
        landuse['landuse'] = 'other'
    
    # Return only necessary columns
    return landuse[['landuse', 'geometry']].copy()


def _save_landuse_cache(landuse: gpd.GeoDataFrame, save_path: Path) -> None:
    """Save land use data to cache file.
    
    Args:
        landuse: Processed land use GeoDataFrame
        save_path: Path to save the cache file
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        landuse.to_file(save_path, driver="GPKG")
        logging.info(f"Saved land use data to {save_path}")
    except Exception as e:
        logging.warning(f"Failed to save cache file {save_path}: {e}")


def compute_landuse_edges(edges_gdf: gpd.GeoDataFrame, 
                         land_gdf: Optional[gpd.GeoDataFrame] = None, 
                         allowed: Optional[Set[str]] = None, 
                         buffer_m: int = LandUseConfig.BUFFER_METERS, 
                         place: Optional[str] = None, 
                         bbox: Optional[Tuple[float, float, float, float]] = None) -> gpd.GeoDataFrame:
    """Add land_use column to edges by buffering and finding nearest land use polygons.
    
    This function performs spatial joining between street edges and land use polygons.
    Each edge is buffered by the specified distance, and the nearest land use polygon
    centroid within the buffer is assigned to that edge.
    
    Args:
        edges_gdf: Street edges with geometry in EPSG:4326
        land_gdf: Pre-loaded land use polygons. If None, loads dynamically
        allowed: Set of allowed land use categories. Defaults to residential/retail/commercial
        buffer_m: Buffer distance in meters for spatial assignment
        place: Place name for dynamic land use generation
        bbox: Bounding box for dynamic land use generation
        
    Returns:
        GeoDataFrame: Copy of edges_gdf with added 'land_use' column
        
    Raises:
        ValueError: If edges_gdf is empty or missing geometry column
    """
    # Input validation
    if edges_gdf.empty:
        raise ValueError("edges_gdf cannot be empty")
    if 'geometry' not in edges_gdf.columns:
        raise ValueError("edges_gdf must have a 'geometry' column")
    
    # Set defaults
    if allowed is None:
        allowed = LandUseConfig.DEFAULT_ALLOWED
    
    # Load land use data
    land_gdf = _get_or_load_landuse_data(land_gdf, place, bbox)
    
    # Handle empty land use data
    if land_gdf.empty:
        return _assign_default_landuse(edges_gdf)
    
    # Perform spatial assignment
    return _perform_spatial_landuse_assignment(edges_gdf, land_gdf, allowed, buffer_m)
    

def _get_or_load_landuse_data(land_gdf: Optional[gpd.GeoDataFrame], 
                             place: Optional[str], 
                             bbox: Optional[Tuple[float, float, float, float]]) -> gpd.GeoDataFrame:
    """Get or load land use data from various sources.
    
    Args:
        land_gdf: Pre-loaded land use data
        place: Place name for dynamic loading
        bbox: Bounding box for dynamic loading
        
    Returns:
        GeoDataFrame: Land use polygons
    """
    if land_gdf is not None:
        return land_gdf
    
    try:
        if place or bbox:
            return get_landuse_polygons(place=place, bbox=bbox)
        else:
            # No dynamic parameters provided, return empty
            logging.warning("No land use data or location parameters provided")
            return gpd.GeoDataFrame(columns=['landuse', 'geometry'], crs="EPSG:4326")
    except Exception as e:
        logging.warning(f"Failed to load land use data: {e}")
        return gpd.GeoDataFrame(columns=['landuse', 'geometry'], crs="EPSG:4326")


def _assign_default_landuse(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign default 'other' land use to all edges.
    
    Args:
        edges_gdf: Street edges GeoDataFrame
        
    Returns:
        GeoDataFrame: Copy with land_use column set to 'other'
    """
    edges_copy = edges_gdf.copy()
    edges_copy["land_use"] = "other"
    return edges_copy


def _perform_spatial_landuse_assignment(edges_gdf: gpd.GeoDataFrame, 
                                       land_gdf: gpd.GeoDataFrame, 
                                       allowed: Set[str], 
                                       buffer_m: int) -> gpd.GeoDataFrame:
    """Perform spatial assignment of land use to edges using buffering and nearest neighbor.
    
    This function:
    1. Reprojects both datasets to metric CRS for accurate buffering
    2. Creates centroids of land use polygons for efficient spatial indexing
    3. For each edge, buffers by specified distance and finds nearest land use centroid
    4. Assigns land use category, defaulting to 'other' if no match found
    
    Args:
        edges_gdf: Street edges in EPSG:4326
        land_gdf: Land use polygons with 'landuse' column
        allowed: Set of allowed land use categories
        buffer_m: Buffer distance in meters
        
    Returns:
        GeoDataFrame: Edges with assigned land_use column
    """
    # Reproject to metric CRS for accurate buffering
    edges_m = edges_gdf.to_crs(epsg=LandUseConfig.CRS_METRIC)
    land_m = land_gdf.to_crs(epsg=LandUseConfig.CRS_METRIC)
    
    # Build spatial index using centroids for efficiency
    centroids = gpd.GeoDataFrame({
        "landuse": land_m["landuse"],
        "geometry": land_m.geometry.centroid
    }, crs=land_m.crs)
    sindex = centroids.sindex
    
    # Assign land use per edge using spatial buffering
    land_assignments = []
    for geom in edges_m.geometry:
        assignment = _find_nearest_landuse(geom, centroids, sindex, allowed, buffer_m)
        land_assignments.append(assignment)
    
    # Create result GeoDataFrame
    edges_copy = edges_gdf.copy()
    edges_copy["land_use"] = pd.Series(land_assignments, index=edges_copy.index)
    edges_copy["land_use"] = edges_copy["land_use"].fillna("other")
    
    return edges_copy


def _find_nearest_landuse(edge_geom, centroids: gpd.GeoDataFrame, 
                         sindex, allowed: Set[str], buffer_m: int) -> Optional[str]:
    """Find nearest allowed land use within buffer distance of edge.
    
    Args:
        edge_geom: Edge geometry
        centroids: Land use centroids with spatial index
        sindex: Spatial index for centroids
        allowed: Set of allowed land use categories
        buffer_m: Buffer distance in meters
        
    Returns:
        str or None: Land use category or None if no match found
    """
    buf = edge_geom.buffer(buffer_m)
    
    # Find candidate centroids whose bounding box intersects buffer
    candidate_indices = list(sindex.intersection(buf.bounds))
    if not candidate_indices:
        return None
    
    # Filter to centroids actually within buffer
    candidates = centroids.iloc[candidate_indices]
    inside_buffer = candidates[candidates.geometry.within(buf)]
    
    # Filter to allowed categories
    allowed_matches = inside_buffer[inside_buffer["landuse"].isin(allowed)]
    
    if allowed_matches.empty:
        return None
    
    # Return first match (closest by spatial index ordering)
    return allowed_matches.iloc[0]["landuse"]


def cleanup_temp_files(max_age_hours: int = LandUseConfig.CACHE_MAX_AGE_HOURS) -> int:
    """Clean up temporary land use cache files older than specified age.
    
    Args:
        max_age_hours: Maximum age in hours before files are deleted
        
    Returns:
        int: Number of files deleted
    """
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    deleted_count = 0
    temp_dir = LandUseConfig.get_temp_dir()
    
    for file_path in temp_dir.glob("*_landuse.gpkg"):
        try:
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
                logging.info(f"Deleted old temp file: {file_path}")
                deleted_count += 1
        except Exception as e:
            logging.warning(f"Failed to process {file_path}: {e}")
    
    return deleted_count


class LandUseError(Exception):
    """Base exception for land use processing errors."""
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


def main():
    """CLI entrypoint for batch processing (DEPRECATED - use API instead).
    
    This function is kept for backward compatibility but the dynamic API
    approach is recommended for new implementations.
    """
    logging.warning("CLI main() function is deprecated. Use the dynamic API approach instead.")
    print("This CLI function has been deprecated. Please use the Flask API with dynamic land use generation.")
    print("Example: curl 'http://localhost:5000/predict?place=Melbourne'")
    return


# Example usage function for testing
def example_usage():
    """Example of how to use the modular land use functions."""
    # Example 1: Get land use for Monaco
    try:
        monaco_landuse = get_landuse_polygons(place="Monaco")
        print(f"Downloaded {len(monaco_landuse)} land use polygons for Monaco")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Process edges (would need actual edge data)
    # edges_with_landuse = compute_landuse_edges(some_edges_gdf, place="Monaco")
    
    # Example 3: Cleanup old cache files
    deleted_count = cleanup_temp_files(max_age_hours=6)
    print(f"Cleaned up {deleted_count} old cache files")


if __name__ == "__main__":
    example_usage()
