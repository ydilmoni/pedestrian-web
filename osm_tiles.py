#!/usr/bin/env python3
"""
OSM tiles helper for ArcGIS integration.
Fetches OSM walk network edges for given bbox and returns as GeoJSON-ready features.
"""

import logging
import os
from typing import List, Dict, Tuple, Optional
import geopandas as gpd
import osmnx as ox
from shapely.geometry import LineString, Point, box as shp_box
import hashlib

# Configure OSMnx
ox.settings.use_cache = True
ox.settings.timeout = 180

logger = logging.getLogger(__name__)


def edges_from_place(place: str, max_features: int = 5000) -> gpd.GeoDataFrame:
    """
    Get pedestrian-accessible edges for a named place.
    Returns GDF with columns: edge_id, osmid, highway, length, geometry (EPSG:4326)
    """
    try:
        G = ox.graph_from_place(place, network_type="walk", simplify=True, retain_all=True)
        gdf = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
        return _postprocess_edges_gdf(gdf, max_features)
    except Exception:
        return gpd.GeoDataFrame(
            columns=["edge_id","osmid","highway","length","geometry"],
            geometry="geometry", crs="EPSG:4326"
        )

def _postprocess_edges_gdf(gdf: gpd.GeoDataFrame, max_features: int) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gpd.GeoDataFrame(
            columns=["edge_id","osmid","highway","length","geometry"],
            geometry="geometry", crs="EPSG:4326"
        )
    keep = [c for c in ["osmid","highway","length","geometry"] if c in gdf.columns]
    gdf = gdf[keep].copy()
    if "osmid" in gdf.columns:
        gdf["osmid"] = gdf["osmid"].apply(lambda v: v[0] if isinstance(v, list) and v else v)
    gdf = gdf.set_crs(4326, allow_override=True)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    gdf = gdf.explode(index_parts=False, ignore_index=True)  # handle MultiLineString
    if "length" not in gdf.columns or gdf["length"].isna().all():
        gdfm = gdf.to_crs(3857)
        gdf["length"] = gdfm.length
    if len(gdf) > max_features:
        gdf = gdf.sample(max_features, random_state=42).reset_index(drop=True)
    gdf = gdf.reset_index(drop=True)
    gdf["edge_id"] = gdf.index.map(lambda i: f"e_{i:07d}")
    return gdf[["edge_id","osmid","highway","length","geometry"]]


def create_edge_id(osmid: str, geometry: LineString) -> str:
    """Create stable edge ID from osmid and geometry hash."""
    geom_wkb = geometry.wkb
    geom_hash = hashlib.md5(geom_wkb).hexdigest()[:12]
    return f"{osmid}_{geom_hash}"


def validate_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """
    Validate and parse bbox string.
    
    Args:
        bbox_str: Comma-separated bbox "west,south,east,north"
    
    Returns:
        Tuple of (west, south, east, north) floats
    
    Raises:
        ValueError: If bbox format is invalid
    """
    try:
        parts = bbox_str.split(',')
        if len(parts) != 4:
            raise ValueError("bbox must have 4 comma-separated values")
        
        west, south, east, north = map(float, parts)
        
        # Validate ranges
        if not (-180 <= west <= 180) or not (-180 <= east <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if not (-90 <= south <= 90) or not (-90 <= north <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if west >= east:
            raise ValueError("West longitude must be less than East longitude")
        if south >= north:
            raise ValueError("South latitude must be less than North latitude")
        
        return west, south, east, north
        
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid bbox format '{bbox_str}': {e}")


def fetch_osm_network(west: float, south: float, east: float, north: float, max_features: int = 15000) -> gpd.GeoDataFrame:
    """
    Fetch OSM walk network for the given bounding box.
    
    Args:
        west, south, east, north: Bounding box coordinates in EPSG:4326
        max_features: Maximum number of features to return
    
    Returns:
        GeoDataFrame with columns: edge_id, osmid, highway, length, geometry
    """
    try:
        logger.info(f"Fetching OSM network for bbox ({west},{south},{east},{north})")
        
        # Download walk network
        G = ox.graph_from_bbox(
            north, south, east, west,
            network_type='walk',
            simplify=True,
            retain_all=False
        )
        
        if len(G.edges) == 0:
            logger.warning("No edges found in bbox")
            return gpd.GeoDataFrame(columns=['edge_id', 'osmid', 'highway', 'length', 'geometry'], crs='EPSG:4326')
        
        # Convert to GeoDataFrame (edges only)
        edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
        
        # Clean and prepare data
        edges_gdf = edges_gdf.reset_index()
        edges_gdf = edges_gdf.to_crs('EPSG:4326')
        edges_gdf = edges_gdf[edges_gdf.geometry.type == 'LineString'].copy()
        
        # Create stable edge IDs
        edges_gdf['osmid_str'] = edges_gdf['osmid'].astype(str)
        edges_gdf['edge_id'] = edges_gdf.apply(
            lambda row: create_edge_id(row['osmid_str'], row.geometry), axis=1
        )
        
        # Calculate length in meters (project to local UTM)
        # For rough length calculation, use a simple approach
        edges_gdf_proj = edges_gdf.to_crs('EPSG:3857')  # Web Mercator for rough calculations
        edges_gdf['length'] = edges_gdf_proj.geometry.length
        
        # Fill missing highway values
        edges_gdf['highway'] = edges_gdf['highway'].fillna('unclassified')
        
        # Keep only required columns
        result = edges_gdf[['edge_id', 'osmid_str', 'highway', 'length', 'geometry']].copy()
        result = result.rename(columns={'osmid_str': 'osmid'})
        
        # Sample if too many features
        if len(result) > max_features:
            logger.info(f"Sampling {max_features} features from {len(result)} available")
            result = result.sample(n=max_features, random_state=42)
        
        logger.info(f"Retrieved {len(result)} OSM walk network edges")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching OSM network: {e}")
        # Return empty GeoDataFrame on error
        return gpd.GeoDataFrame(columns=['edge_id', 'osmid', 'highway', 'length', 'geometry'], crs='EPSG:4326')


def edges_for_bbox(w: float, s: float, e: float, n: float, max_features: int = 5000):
    """
    Return OSM walk edges for bbox as GeoDataFrame with columns:
      edge_id, osmid, highway, length, geometry (EPSG:4326)
    """
    import geopandas as gpd, osmnx as ox
    try:
        G = ox.graph_from_bbox(north=n, south=s, east=e, west=w,
                               network_type="walk", simplify=True, retain_all=True)
        gdf = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
        out = _postprocess_edges_gdf(gdf, max_features)
        if not out.empty:
            return out

        # fallback: broader pedestrian filter
        cf = ('["highway"]["area"!~"yes"]'
              '["highway"~"footway|path|pedestrian|living_street|residential|unclassified|service|track|steps|tertiary|secondary|primary"]'
              '["access"!~"private"]')
        G2 = ox.graph_from_bbox(north=n, south=s, east=e, west=w,
                                custom_filter=cf, simplify=True, retain_all=True)
        g2 = ox.graph_to_gdfs(G2, nodes=False, fill_edge_geometry=True)
        out2 = _postprocess_edges_gdf(g2, max_features)
        if not out2.empty:
            return out2

        # final fallback: polygon-based query with padding
        try:
            poly = shp_box(w, s, e, n).buffer(0.001)  # ~100m pad
            G3 = ox.graph_from_polygon(poly, network_type="walk",
                                       simplify=True, retain_all=True, truncate_by_edge=True)
            g3 = ox.graph_to_gdfs(G3, nodes=False, fill_edge_geometry=True)
            out3 = _postprocess_edges_gdf(g3, max_features)
            if not out3.empty:
                return out3
        except Exception:
            # Polygon fallback failed, try point-based approach
            pass

        # ultimate fallback: point-based query with clipping
        try:
            center = Point((w+e)/2.0, (s+n)/2.0)
            G4 = ox.graph_from_point((center.y, center.x), dist=4000, network_type="walk",
                                     simplify=True, retain_all=True)
            g4 = ox.graph_to_gdfs(G4, nodes=False, fill_edge_geometry=True)
            bbox_poly = shp_box(w, s, e, n)
            g4 = g4.set_crs(4326, allow_override=True)
            g4 = g4[g4.geometry.intersects(bbox_poly)]
            return _postprocess_edges_gdf(g4, max_features)
        except Exception:
            # Even point fallback failed
            pass

    except Exception:
        return gpd.GeoDataFrame(
            columns=["edge_id","osmid","highway","length","geometry"],
            geometry="geometry", crs="EPSG:4326"
        )


def get_network_geojson(bbox_str: str, max_features: int = 15000) -> Dict:
    """
    Get OSM walk network as GeoJSON for the given bounding box.
    
    Args:
        bbox_str: Comma-separated bbox "west,south,east,north"
        max_features: Maximum number of features to return
    
    Returns:
        GeoJSON FeatureCollection dict
    
    Raises:
        ValueError: If bbox is invalid
    """
    # Validate bbox
    west, south, east, north = validate_bbox(bbox_str)
    
    # Fetch network
    edges_gdf = fetch_osm_network(west, south, east, north, max_features)
    
    if len(edges_gdf) == 0:
        return {
            "type": "FeatureCollection",
            "features": []
        }
    
    # Convert to GeoJSON format
    geojson_dict = {
        "type": "FeatureCollection",
        "features": []
    }
    
    for _, row in edges_gdf.iterrows():
        feature = {
            "type": "Feature",
            "properties": {
                "edge_id": str(row['edge_id']),
                "osmid": str(row['osmid']),
                "highway": str(row['highway']),
                "length": float(row['length'])
            },
            "geometry": {
                "type": "LineString",
                "coordinates": list(row.geometry.coords)
            }
        }
        geojson_dict["features"].append(feature)
    
    logger.info(f"Created GeoJSON with {len(geojson_dict['features'])} features")
    return geojson_dict