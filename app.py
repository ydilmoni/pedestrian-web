#!/usr/bin/env python3
# pedestrian-api/app.py

# --- ensure sibling modules (like osm_tiles.py) are importable in Render ---
import os, sys
# Add both the directory containing this file and the current working directory
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
# ---------------------------------------------------------------------------

# Optional orjson shim (safe on Python 3.11/3.13 with or without orjson installed)
try:
    import orjson as _orjson
    def fast_dumps(obj) -> bytes:
        return _orjson.dumps(obj)  # returns bytes
except Exception:
    import json as _json
    def fast_dumps(obj) -> bytes:
        # compact, UTF-8 JSON bytes (approximate orjson defaults)
        return _json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

from flask import Flask, request, jsonify, Response, send_file, after_this_request
from flask_cors import CORS
import tempfile
import pandas as pd
import logging
import time
import datetime as dt
from datetime import datetime
from typing import Optional, Tuple
from catboost import CatBoostClassifier, Pool
import math
import json
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.geometry import shape
import sqlite3
import textwrap
import re

# Import the unified feature pipeline
from feature_engineering.feature_pipeline import (
    run_feature_pipeline,
    prepare_model_features,
    PipelineError,
    PipelineConfig
)

# Import OSM tiles helper for ArcGIS integration
try:
    from osm_tiles import edges_for_bbox, edges_from_place
except ImportError as e:
    # Fallback: search multiple locations for osm_tiles.py
    import importlib.util
    import glob
    
    # Debug info for troubleshooting
    print(f"DEBUG: __file__ = {__file__}")
    print(f"DEBUG: os.path.dirname(__file__) = {os.path.dirname(__file__)}")
    print(f"DEBUG: os.getcwd() = {os.getcwd()}")
    print(f"DEBUG: sys.path = {sys.path[:3]}...")  # First 3 entries
    
    # Search locations in order of preference
    search_paths = [
        os.path.join(os.path.dirname(__file__), 'osm_tiles.py'),           # Same dir as app.py
        os.path.join(os.getcwd(), 'osm_tiles.py'),                         # Current working dir
        os.path.join(os.getcwd(), 'pedestrian-api', 'osm_tiles.py'),       # CWD/pedestrian-api/
        os.path.join(os.path.dirname(__file__), '..', 'osm_tiles.py'),     # Parent dir
    ]
    
    # Also search with glob pattern
    glob_patterns = [
        '**/osm_tiles.py',
        'pedestrian-api/osm_tiles.py',
        './osm_tiles.py'
    ]
    
    for pattern in glob_patterns:
        try:
            found_files = glob.glob(pattern, recursive=True)
            search_paths.extend(found_files)
        except:
            pass
    
    print(f"DEBUG: Searching for osm_tiles.py in {len(search_paths)} locations...")
    
    osm_tiles_path = None
    for path in search_paths:
        print(f"DEBUG: Checking {path}")
        if os.path.exists(path):
            osm_tiles_path = path
            print(f"DEBUG: Found osm_tiles.py at {path}")
            break
    
    if osm_tiles_path:
        spec = importlib.util.spec_from_file_location("osm_tiles", osm_tiles_path)
        osm_tiles = importlib.util.module_from_spec(spec)
        sys.modules["osm_tiles"] = osm_tiles
        spec.loader.exec_module(osm_tiles)
        edges_for_bbox = osm_tiles.edges_for_bbox
        edges_from_place = osm_tiles.edges_from_place
        print(f"DEBUG: Successfully loaded osm_tiles from {osm_tiles_path}")
    else:
        # List all files in current directory for debugging
        try:
            files = os.listdir(os.path.dirname(__file__) or '.')
            print(f"DEBUG: Files in app directory: {files}")
        except:
            pass
        try:
            files = os.listdir(os.getcwd())
            print(f"DEBUG: Files in working directory: {files}")
        except:
            pass
        raise ImportError(f"Could not import osm_tiles: {e}. Searched {len(search_paths)} locations but file not found.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

app = Flask(__name__)

ALLOWED_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",  # Vite dev
    "http://localhost:3000", "http://127.0.0.1:3000",  # CRA dev
    "http://localhost:3001", "http://127.0.0.1:3001",  # Experience Builder dev
    "http://localhost:8080", "http://127.0.0.1:8080",  # Alternative port
    "https://experience.arcgis.com",                     # published Experience origin
    "https://ariel-surveying.maps.arcgis.com",           # org portal (useful for previews/embeds)
    "https://pedestrian-api.onrender.com",               # current API hosting
    "https://<your-site>.netlify.app"                   # alt hosting
]

CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

def json_response(obj, status: int = 200) -> Response:
    return Response(fast_dumps(obj), status=status, mimetype="application/json")

# GPKG Export Helper Functions
def _slug(text: str) -> str:
    """Convert text to URL-safe slug."""
    import re
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    return text.strip('-')

def _proba_columns(proba, start_label: int = 1):
    """Generate probability columns with proper naming and proba_top."""
    if not proba:
        return {}
    cols = {f"proba_{i}": float(p) for i, p in enumerate(proba, start=start_label)}
    try:
        cols["proba_top"] = float(max(proba))
    except Exception:
        pass
    return cols

def _items_to_geoms(items: list, place: str = None) -> gpd.GeoDataFrame:
    """Convert prediction items to geometries using OSM network."""
    if not items:
        return gpd.GeoDataFrame()
    
    # Extract edge_ids from items
    edge_ids = [item.get('edge_id') for item in items if item.get('edge_id')]
    
    if not edge_ids or not place:
        # Create dummy point geometries if no place or edge_ids
        geoms = [Point(0, 0) for _ in items]
        return gpd.GeoDataFrame(items, geometry=geoms, crs=4326)
    
    try:
        # Get network data for the place
        network_gdf = edges_from_place(place, max_features=10000)
        
        # Filter to matching edge_ids
        matching_edges = network_gdf[network_gdf['edge_id'].isin(edge_ids)]
        
        # Create result GeoDataFrame
        result_data = []
        for item in items:
            edge_id = item.get('edge_id')
            matching_row = matching_edges[matching_edges['edge_id'] == edge_id]
            
            if len(matching_row) > 0:
                geom = matching_row.iloc[0].geometry
                result_data.append({**item, 'geometry': geom})
            else:
                # Fallback to point geometry
                result_data.append({**item, 'geometry': Point(0, 0)})
        
        return gpd.GeoDataFrame(result_data, crs=4326)
        
    except Exception as e:
        logging.warning(f"Failed to get geometries for place {place}: {e}")
        # Fallback to point geometries
        geoms = [Point(0, 0) for _ in items]
        return gpd.GeoDataFrame(items, geometry=geoms, crs=4326)

def _items_to_geoms_list(items, place: str | None = None, bbox: str | None = None):
    """
    Turn items into shapely geometries list.
    Priority:
      1) item['geometry'] present (shapely or GeoJSON-like) -> use it
      2) else, if available, map by edge_id from the base network
    """
    geoms = []
    edge_geom_map = {}
    
    # Try to load base network for geometry mapping
    try:
        edges_gdf = edges_from_place(place) if place else None
        if edges_gdf is not None and not edges_gdf.empty:
            if getattr(edges_gdf, "crs", None) and str(edges_gdf.crs).upper() != "EPSG:4326":
                edges_gdf = edges_gdf.to_crs("EPSG:4326")
            id_col = "edge_id" if "edge_id" in edges_gdf.columns else None
            if id_col:
                edge_geom_map = dict(zip(edges_gdf[id_col], edges_gdf.geometry))
    except Exception:
        pass

    for it in items:
        g = it.get("geometry")
        if g is not None:
            # already shapely?
            if hasattr(g, "geom_type"):
                geoms.append(g)
            else:
                try:
                    geoms.append(shape(g))
                except Exception:
                    geoms.append(None)
        else:
            eid = it.get("edge_id")
            geoms.append(edge_geom_map.get(eid))
    return geoms

def _build_predictions_gdf_new(items, out, place: str | None = None, bbox: str | None = None, crs_epsg: int = 4326):
    """Build predictions GeoDataFrame with proper field structure."""
    preds = out.get("predictions", [])
    geoms = _items_to_geoms_list(items, place=place, bbox=bbox)

    rows = []
    for p in preds:
        base = {
            "edge_id": p.get("edge_id"),
            "volume_class": p.get("volume_class"),  # expected 1..5
        }
        base.update(_proba_columns(p.get("proba"), start_label=1))
        feats = p.get("features") or {}
        for k in ["Hour","betweenness","closeness","is_weekend","time_of_day","land_use","highway"]:
            if k in feats:
                base[k] = feats[k]
        # force strings for categorical fields
        for k in ("time_of_day","land_use","highway"):
            if k in base and base[k] is not None:
                base[k] = str(base[k])
        rows.append(base)

    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=f"EPSG:{crs_epsg}")
    gdf = gdf[~gdf.geometry.isna()].copy()
    if gdf.empty:
        return gdf

    # fix invalid geometries for QGIS
    try:
        gdf.geometry = gdf.geometry.apply(make_valid)
    except Exception:
        gdf.geometry = gdf.buffer(0)
    return gdf

def _build_predictions_gdf(predictions: dict, place: str = None) -> gpd.GeoDataFrame:
    """Build GeoDataFrame from prediction results."""
    items = predictions.get('predictions', [])
    if not items:
        return gpd.GeoDataFrame()
    
    # Convert items to geometries
    gdf = _items_to_geoms(items, place)
    
    # Add probability columns
    if items and 'proba' in items[0]:
        num_classes = len(items[0]['proba'])
        proba_cols = _proba_columns(num_classes)
        
        for i, col in enumerate(proba_cols):
            gdf[col] = [item.get('proba', [0] * num_classes)[i] for item in items]
    
    # Ensure required columns exist
    if 'volume_class' not in gdf.columns:
        gdf['volume_class'] = [item.get('volume_class', 0) for item in items]
    
    return gdf

def clean_geojson(geojson_dict):
    """Clean GeoJSON by replacing NaN values with null."""
    import json
    import math
    
    def clean_value(obj):
        if isinstance(obj, dict):
            return {k: clean_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_value(item) for item in obj]
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        else:
            return obj
    
    return clean_value(geojson_dict)

# QGIS styling helper functions
def _gpkg_geom_column(gpkg_path: str, table: str) -> str:
    """Get the geometry column name for a table in the GPKG."""
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.execute("SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?", (table,))
        row = cur.fetchone()
        return row[0] if row and row[0] else "geom"
    finally:
        conn.close()

def _ensure_qgis_layer_styles(conn: sqlite3.Connection):
    """Ensure the layer_styles table exists in the GPKG for QGIS styling."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS layer_styles (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      f_table_catalog TEXT,
      f_table_schema TEXT,
      f_table_name TEXT,
      f_geometry_column TEXT,
      styleName TEXT,
      styleQML TEXT,
      styleSLD TEXT,
      useAsDefault INTEGER,
      description TEXT,
      owner TEXT,
      ui BLOB,
      update_time TEXT
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_layer_styles_table ON layer_styles(f_table_name);")

def _install_qgis_style(gpkg_path: str, table: str, qml_xml: str, style_name: str = "default", use_default: int = 1):
    """Install a QGIS style into the GPKG layer_styles table."""
    geom_col = _gpkg_geom_column(gpkg_path, table)
    conn = sqlite3.connect(gpkg_path)
    try:
        _ensure_qgis_layer_styles(conn)
        conn.execute("UPDATE layer_styles SET useAsDefault=0 WHERE f_table_name=?;", (table,))
        conn.execute("""
            INSERT INTO layer_styles
            (f_table_catalog, f_table_schema, f_table_name, f_geometry_column,
             styleName, styleQML, styleSLD, useAsDefault, description, owner, ui, update_time)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
        """, ("", "", table, geom_col, style_name, qml_xml, None, use_default, "auto-installed by API", "api", None))
        conn.commit()
    finally:
        conn.close()

def _qml_predictions_categorized() -> str:
    """Generate QML style for categorized volume_class with graduated line widths."""
    categories = [
        ("1", "1 - Very Low", "#4c78a8", "0.40"),
        ("2", "2 - Low",      "#72b7b2", "0.70"),
        ("3", "3 - Medium",   "#e6b8a2", "1.10"),
        ("4", "4 - High",     "#f58518", "1.60"),
        ("5", "5 - Very High","#e45756", "2.20"),
    ]
    cats_xml, syms_xml = [], []
    for i,(val,label,color,width) in enumerate(categories):
        cats_xml.append(f'<category symbol="{i}" value="{val}" label="{label}"/>')
        syms_xml.append(textwrap.dedent(f"""
        <symbol name="{i}" type="line" alpha="1">
          <layer class="SimpleLine" enabled="1" pass="0" locked="0">
            <prop k="capstyle" v="round"/>
            <prop k="joinstyle" v="round"/>
            <prop k="line_color" v="{color}"/>
            <prop k="line_style" v="solid"/>
            <prop k="line_width" v="{width}"/>
            <prop k="line_width_unit" v="MM"/>
          </layer>
        </symbol>
        """).strip())
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3" styleCategories="Symbology">
  <renderer-v2 type="categorizedSymbol" attr="volume_class" symbollevels="0" enableorderby="0" forceraster="0">
    <categories>{' '.join(cats_xml)}</categories>
    <symbols>{' '.join(syms_xml)}</symbols>
  </renderer-v2>
  <layerGeometryType>1</layerGeometryType>
</qgis>
"""

# Load the pre-trained CatBoost model (cb-loco model trained on Dublin, Melbourne, NYC, tested on Zurich)
MODEL_PATH = os.getenv(
    "MODEL_PATH", 
    os.path.join(os.path.dirname(__file__), "models", "cb_loco_train_dublin_melbourne_nyc_test_zurich.cbm")
)

try:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    logging.info(f"Successfully loaded CatBoost model from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Failed to load model from {MODEL_PATH}: {e}")
    model = None

# Use configuration from pipeline
FEATS = PipelineConfig.FEATURE_COLUMNS
CAT_COLS = PipelineConfig.CATEGORICAL_COLUMNS


def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    return gdf

def _to_meters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project to WebMercator for length/geometry ops in meters."""
    return gdf.to_crs(3857)

def _apply_edits(base_edges: gpd.GeoDataFrame, edits: list, snap_tol_m: float = 8.0) -> gpd.GeoDataFrame:
    """
    Apply simple edit ops to base edges:
      - {"op":"delete","edge_id":...}
      - {"op":"add","geometry": LineString GeoJSON}
      - {"op":"reshape","edge_id":...,"geometry": LineString GeoJSON}
    Snap endpoints of added/reshaped lines to nearest base endpoints within snap_tol_m.
    """
    if base_edges.empty:
        base_edges = gpd.GeoDataFrame(columns=["edge_id","osmid","highway","length","geometry"], geometry="geometry", crs=4326)

    # Work in meters for snapping
    g_m = _to_meters(_to_wgs84(base_edges)).copy()
    # Precompute endpoint index (naive, small bbox)
    endpoints = []
    for idx, geom in g_m.geometry.items():
        if geom is None or geom.is_empty: continue
        try:
            xs, ys = list(geom.coords)[0], list(geom.coords)[-1]
            endpoints.append((idx, Point(xs[0], xs[1])))
            endpoints.append((idx, Point(ys[0], ys[1])))
        except Exception:
            pass
    ep_gdf = gpd.GeoDataFrame({"edge_row":[i for i,_ in endpoints]}, geometry=[p for _,p in endpoints], crs=3857)

    def _snap_ls(ls: LineString) -> LineString:
        if ep_gdf.empty: return ls
        # snap both ends if within tol
        coords = list(ls.coords)
        P0 = Point(coords[0]); P1 = Point(coords[-1])
        for i,P in [(0,P0),(len(coords)-1,P1)]:
            dists = ep_gdf.distance(P)
            j = int(dists.idxmin()) if len(dists) else None
            if j is not None and dists.loc[j] <= snap_tol_m:
                # move coord to nearest endpoint
                coords[i] = (ep_gdf.loc[j].geometry.x, ep_gdf.loc[j].geometry.y)
        return LineString(coords)

    # Build mutable copy in meters
    out = g_m.copy()

    for e in edits or []:
        op = (e.get("op") or e.get("action") or "").lower()
        if op == "delete" and e.get("edge_id"):
            out = out[out["edge_id"] != e["edge_id"]].copy()

        elif op in ("add","reshape"):
            geom_geojson = e.get("geometry")
            if not geom_geojson or geom_geojson.get("type") != "LineString":
                continue
            # incoming geometry is WGS84 → to meters
            new_wgs = gpd.GeoSeries([LineString(geom_geojson["coordinates"])], crs=4326).to_crs(3857).iloc[0]
            new_ls = _snap_ls(new_wgs)
            # basic attrs
            hw = str(e.get("highway") or "unclassified")
            row = {
                "edge_id": e.get("edge_id") or f"e_add_{len(out)+1:06d}",
                "osmid": e.get("osmid") or None,
                "highway": hw,
                "length": new_ls.length,
                "geometry": new_ls
            }
            if op == "reshape" and e.get("edge_id"):
                out = out[out["edge_id"] != e["edge_id"]].copy()
            out = gpd.GeoDataFrame(pd.concat([out, gpd.GeoDataFrame([row], geometry="geometry", crs=3857)], ignore_index=True))
        else:
            # ignore unknown ops
            pass

    # back to WGS84 and recompute length (meters)
    out_wgs = out.to_crs(4326)
    out_m   = out  # already meters
    out_wgs["length"] = out_m.length
    # normalize columns
    keep = [c for c in ["edge_id","osmid","highway","length","geometry"] if c in out_wgs.columns]
    out_wgs = out_wgs[keep]
    return out_wgs

def _graph_from_edges(edges_wgs: gpd.GeoDataFrame) -> nx.Graph:
    """
    Build an undirected graph where nodes are rounded endpoints (to ~1e-6 deg),
    edges carry length and a reference to edge_id(s). Good enough for centrality.
    """
    if edges_wgs.empty:
        return nx.Graph()
    E = _to_meters(edges_wgs)
    G = nx.Graph()
    for _, row in E.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        coords = list(geom.coords)
        # Ensure coordinates are hashable tuples of floats
        def coord_to_tuple(coord):
            # Handle various coordinate types (lists, tuples, numpy arrays)
            if hasattr(coord[0], 'item'):  # numpy scalar
                return (round(coord[0].item(), 3), round(coord[1].item(), 3))
            else:
                return (round(float(coord[0]), 3), round(float(coord[1]), 3))
        
        u = coord_to_tuple(coords[0])
        v = coord_to_tuple(coords[-1])
        wlen = float(geom.length)
        eid  = row.get("edge_id")
        if not G.has_edge(u, v):
            G.add_edge(u, v, length=wlen, edge_ids=set())
        G[u][v]["edge_ids"].add(eid)
    return G

def _centrality_features(G: nx.Graph, k: int = 60) -> dict:
    """
    Compute fast approx centrality:
      - edge betweenness (k-sampled)
      - node closeness → edge_closeness = avg of endpoints
    Returns dict: edge_id -> {"edge_betweenness":..., "edge_closeness":...}
    """
    if G.number_of_edges() == 0:
        return {}
    # k-sample nodes for betweenness
    nodes = list(G.nodes())
    # Debug: check node types
    for i, node in enumerate(nodes[:3]):  # Check first 3 nodes
        logging.info(f"Node {i}: {node} (type: {type(node)})")
    if len(nodes) <= k:
        k_nodes = nodes
    else:
        rng = np.random.default_rng(42)
        k_nodes = list(rng.choice(nodes, size=k, replace=False))
    eb = nx.edge_betweenness_centrality_subset(G, sources=k_nodes, targets=k_nodes, weight="length", normalized=True)
    # closeness per node (distance-weighted)
    cn = nx.closeness_centrality(G, distance="length")
    # map to edges by averaging endpoint closeness
    out = {}
    for (u, v), b in eb.items():
        ids = G[u][v]["edge_ids"]
        c = 0.5*(cn.get(u, 0.0) + cn.get(v, 0.0))
        for eid in ids:
            out[eid] = {"edge_betweenness": float(b), "edge_closeness": float(c)}
    return out

def _highway_ordinal(hw) -> float:
    # Handle both string and list cases
    if isinstance(hw, list):
        hw = hw[0] if hw else ""
    hw = str(hw or "").lower()
    if hw in ("footway","path","steps","pedestrian"): return 0.5
    if hw in ("residential","living_street","service","unclassified"): return 1.0
    if hw in ("tertiary","tertiary_link"): return 2.0
    if hw in ("secondary","secondary_link"): return 3.0
    if hw in ("primary","primary_link"): return 4.0
    return 1.5

def _build_features(edges_wgs: gpd.GeoDataFrame, cen: dict, feature_order: list | None) -> pd.DataFrame:
    """
    Build a feature table aligned with FEATURE_ORDER if provided.
    Minimal set if not provided: edge_length_m, edge_betweenness, edge_closeness, highway_ord
    """
    df = edges_wgs.copy()
    df["edge_length_m"] = df["length"].astype(float)
    df["edge_betweenness"] = df["edge_id"].map(lambda i: cen.get(i, {}).get("edge_betweenness", 0.0))
    df["edge_closeness"]   = df["edge_id"].map(lambda i: cen.get(i, {}).get("edge_closeness", 0.0))
    df["highway_ord"]      = df["highway"].map(_highway_ordinal)

    if feature_order:
        # create missing columns as zeros, keep only order
        for col in feature_order:
            if col not in df.columns:
                df[col] = 0.0
        X = df[feature_order].copy()
    else:
        X = df[["edge_length_m","edge_betweenness","edge_closeness","highway_ord"]].copy()
    X = X.replace([np.inf,-np.inf], 0).fillna(0)
    return df, X


def validate_request_params(place: Optional[str], bbox_str: Optional[str], date: Optional[str]) -> Tuple[Optional[str], Optional[Tuple[float, float, float, float]], Optional[str]]:
    """Validate and parse request parameters.
    
    Args:
        place: Place name parameter
        bbox_str: Bounding box string parameter
        date: Date string parameter
        
    Returns:
        tuple: (validated_place, parsed_bbox, validated_date)
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Validate place or bbox
    if not place and not bbox_str:
        raise ValueError("Either 'place' or 'bbox' parameter is required")
    
    # Parse and validate bbox
    bbox = None
    if bbox_str:
        try:
            bbox_parts = [float(x.strip()) for x in bbox_str.split(",")]
            if len(bbox_parts) != 4:
                raise ValueError("Bbox must contain exactly 4 coordinates")
            bbox = tuple(bbox_parts)
            
            # Validate coordinate ranges
            minx, miny, maxx, maxy = bbox
            if not (-180 <= minx <= 180 and -180 <= maxx <= 180):
                raise ValueError("Longitude values must be between -180 and 180")
            if not (-90 <= miny <= 90 and -90 <= maxy <= 90):
                raise ValueError("Latitude values must be between -90 and 90")
            if minx >= maxx or miny >= maxy:
                raise ValueError("Invalid bbox: min values must be less than max values")
                
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid bbox format: {e}")
    
    # Validate place
    if place and not isinstance(place, str):
        raise ValueError("Place parameter must be a string")
    
    # Validate date
    if date:
        try:
            pd.to_datetime(date)
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")
    
    return place, bbox, date


# FEATURE_COLUMNS and CATEGORICAL_COLUMNS for batch inference
FEATURE_COLUMNS = ["betweenness","closeness","Hour","is_weekend","time_of_day","land_use","highway"]
CATEGORICAL_COLUMNS = ["time_of_day","land_use","highway"]

def _prepare_df(items):
    """Prepare DataFrame for batch prediction."""
    import numpy as np
    df = pd.DataFrame(items)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    df = df[FEATURE_COLUMNS]
    for c in CATEGORICAL_COLUMNS:
        df[c] = df[c].astype(str)
    for c in ["betweenness","closeness","Hour","is_weekend"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.isna().any().any():
        raise ValueError("NaNs in features after coercion.")
    return df

def predict_with_model(items):
    """Batch prediction with CatBoost model."""
    df = _prepare_df(items)
    y_pred = model.predict(df)
    y_proba = model.predict_proba(df)
    n_classes = y_proba.shape[1] if hasattr(y_proba, "shape") and len(y_proba.shape)==2 else 1

    results = []
    for i, row in enumerate(items):
        try:
            cls = int(y_pred[i])
        except Exception:
            cls = y_pred[i]
        probs = y_proba[i].tolist() if n_classes > 1 else [1.0]
        results.append({
            "edge_id": row.get("edge_id"),
            "features": {k: row.get(k) for k in FEATURE_COLUMNS},
            "volume_class": cls,
            "proba": probs
        })
    return {"predictions": results, "classes": list(range(n_classes))}

def create_prediction_response(features_gdf, predictions, metadata) -> dict:
    """Create structured prediction response.
    
    Args:
        features_gdf: GeoDataFrame with features and predictions
        predictions: Model predictions array
        metadata: Pipeline processing metadata
        
    Returns:
        dict: Structured JSON response
    """
    # Add predictions to features
    features_gdf = features_gdf.copy()
    features_gdf["volume_bin"] = predictions.astype(int)
    
    # Get sample features for response (convert numpy types to native Python types)
    sample_features = {}
    if len(features_gdf) > 0:
        sample_row = features_gdf.iloc[0]
        for col in FEATS:
            if col in sample_row:
                val = sample_row[col]
                # Convert numpy/pandas types to JSON-serializable types
                if hasattr(val, 'item'):  # numpy scalars
                    sample_features[col] = val.item()
                elif pd.api.types.is_integer_dtype(type(val)):
                    sample_features[col] = int(val)
                elif pd.api.types.is_float_dtype(type(val)):
                    sample_features[col] = float(val)
                else:
                    sample_features[col] = str(val)
    
    # Convert metadata values to JSON-serializable types
    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif pd.api.types.is_integer_dtype(type(obj)):
            return int(obj)
        elif pd.api.types.is_float_dtype(type(obj)):
            return float(obj)
        else:
            return obj
    
    # Create response
    response = {
        "success": True,
        "location": metadata["location"],
        "timestamp": metadata["timestamp"],
        "processing_time": round(float(metadata["processing_time"]), 2),
        "network_stats": make_json_serializable(metadata["network_stats"]),
        "sample_prediction": {
            "volume_bin": int(predictions[0]) if len(predictions) > 0 else None,
            "features": sample_features
        },
        "validation": make_json_serializable(metadata["validation"]),
        "geojson": features_gdf.__geo_interface__
    }
    
    return response


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint."""
    return jsonify({"pong": True})


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "pedestrian-api", "port": 8000})


@app.route("/base-network", methods=["GET"])
def base_network():
    """Get OSM walk network edges for ArcGIS."""
    try:
        place = request.args.get("place")
        bbox = request.args.get("bbox")
        max_features = request.args.get("max_features", 5000, type=int)
        
        if place:
            gdf = edges_from_place(place, max_features)
            clean_data = clean_geojson(gdf.__geo_interface__)
            return json_response(clean_data)
            
        if not bbox:
            return jsonify({"error":"provide ?place=... or ?bbox=w,s,e,n"}), 400
            
        try:
            w, s, e, n = map(float, bbox.split(","))
        except Exception:
            return jsonify({"error":"bbox must be 'west,south,east,north'"}), 400
            
        gdf = edges_for_bbox(w, s, e, n, max_features)
        clean_data = clean_geojson(gdf.__geo_interface__)
        return json_response(clean_data)
        
    except Exception as e:
        logging.error(f"Error in base_network: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/simulate", methods=["POST"])
def simulate():
    """
    PLACE-FIRST simulate:
      payload = {
        "place": "Tel Aviv, Israel",         # preferred
        "edits": [ { "op": "...", ... } ],
        "max_features": 8000                 # optional
      }
    Fallback if place missing:
      payload = { "bbox": "w,s,e,n", "edits":[...], "max_features": ... }
    Returns: GeoJSON FeatureCollection with pred_before, pred_after, delta.
    """
    try:
        payload = request.get_json(silent=True) or {}
        place = payload.get("place", None)
        bbox  = payload.get("bbox", None)
        edits = payload.get("edits", [])
        max_features = int(payload.get("max_features", 8000))
        if not isinstance(edits, list):
            return jsonify({"error":"edits must be a list"}), 400

        # -------------------------------
        # 1) Baseline edges (PLACE first)
        # -------------------------------
        if place and isinstance(place, str) and place.strip():
            base_edges = edges_from_place(place.strip(), max_features=max_features)
        elif bbox:
            try:
                w,s,e,n = [float(x) for x in str(bbox).split(",")]
                assert w < e and s < n
            except Exception:
                return jsonify({"error":"bbox must be 'west,south,east,north' (WGS84)"}), 400
            base_edges = edges_for_bbox(w,s,e,n, max_features=max_features)
        else:
            return jsonify({"error":"provide 'place' or 'bbox'"}), 400

        base_edges = base_edges[["edge_id","osmid","highway","length","geometry"]].copy()

        # 2) Apply edits → scenario edges
        scen_edges = _apply_edits(base_edges, edits, snap_tol_m=8.0)

        # 3) Skip complex centrality for now, use simple proxy
        # G0 = _graph_from_edges(base_edges)
        # G1 = _graph_from_edges(scen_edges)
        # cen0 = _centrality_features(G0, k=60)
        # cen1 = _centrality_features(G1, k=60)
        cen0 = {}  # Use empty centrality for now
        cen1 = {}

        # 4) Features aligned to FEATURE_ORDER if present
        feature_order = globals().get("FEATURE_ORDER", None)
        df0, X0 = _build_features(base_edges, cen0, feature_order)
        df1, X1 = _build_features(scen_edges, cen1, feature_order)

        # 5) Predict with CatBoost if loaded, else centrality proxy
        m = globals().get("model", None)
        if m is None:
            df0["pred"] = df0["edge_betweenness"]*1000
            df1["pred"] = df1["edge_betweenness"]*1000
        else:
            try:
                p0 = m.predict(X0.values)
                p1 = m.predict(X1.values)
            except Exception:
                p0 = np.zeros(len(X0), dtype=float)
                p1 = np.zeros(len(X1), dtype=float)
            df0["pred"] = p0.astype(float)
            df1["pred"] = p1.astype(float)

        # 6) Join BEFORE vs AFTER by edge_id
        left = df0[["edge_id","pred"]].rename(columns={"pred":"pred_before"})
        right = df1[["edge_id","pred"]].rename(columns={"pred":"pred_after"})
        merged = right.merge(left, on="edge_id", how="left")
        merged["pred_before"] = merged["pred_before"].fillna(0.0)
        merged["delta"] = merged["pred_after"] - merged["pred_before"]

        # 7) Attach props, return GeoJSON
        scen = scen_edges.merge(merged[["edge_id","pred_before","pred_after","delta"]],
                                on="edge_id", how="left").fillna({"pred_before":0.0,"pred_after":0.0,"delta":0.0})
        
        # Clean NaN values before serialization
        clean_geojson_data = clean_geojson(scen.__geo_interface__)
        return json_response(clean_geojson_data)

    except Exception as e:
        logging.exception("simulate failed")
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """Batch prediction endpoint for CatBoost model."""
    payload = request.get_json(silent=True) or {}
    items = payload.get("items", [])
    if not items:
        return jsonify({"error":"Provide JSON with 'items': [...]"}), 400
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
        
    try:
        out = predict_with_model(items)
        return jsonify(out), 200
    except Exception as e:
        logging.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["GET"])
def predict():
    """Predict pedestrian volume bins (1-5) for the given place and date.
    
    Query Parameters:
        place (str, optional): Place name (e.g., "Monaco", "Tel Aviv")
        bbox (str, optional): Bounding box as "minx,miny,maxx,maxy"
        date (str, optional): ISO timestamp (defaults to current time)
        
    Returns:
        JSON response with predictions and metadata
    """
    request_start = time.time()
    
    # Get request parameters
    place = request.args.get("place")
    date = request.args.get("date")
    bbox_str = request.args.get("bbox")
    
    logging.info(f"Received prediction request: place={place}, bbox={bbox_str}, date={date}")
    
    try:
        # 1. Validate request parameters
        place, bbox, date = validate_request_params(place, bbox_str, date)
        
        # 2. Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not available",
                "code": 503,
                "details": "CatBoost model failed to load at startup"
            }), 503
        
        # 3. Run feature extraction pipeline
        features_gdf, pipeline_metadata = run_feature_pipeline(
            place=place,
            bbox=bbox,
            timestamp=date
        )
        
        # 4. Prepare features for model
        model_features = prepare_model_features(features_gdf)
        
        # 5. Make predictions
        logging.info(f"Making predictions for {len(model_features)} edges")
        
        # Get categorical feature indices
        cat_feature_indices = [model_features.columns.get_loc(col) for col in CAT_COLS if col in model_features.columns]
        
        # Ensure categorical features are strings
        for col in CAT_COLS:
            if col in model_features.columns:
                model_features[col] = model_features[col].astype(str)
        
        # Create CatBoost Pool and predict
        pool = Pool(model_features, cat_features=cat_feature_indices)
        predictions = model.predict(pool)
        
        # 6. Create response
        total_time = time.time() - request_start
        pipeline_metadata["total_request_time"] = total_time
        
        response = create_prediction_response(features_gdf, predictions, pipeline_metadata)
        
        logging.info(f"Prediction completed in {total_time:.2f}s for {len(predictions)} edges")
        
        return jsonify(response)
        
    except PipelineError as e:
        logging.error(f"Pipeline error: {e.message}")
        return jsonify(e.to_dict()), e.code
        
    except ValueError as e:
        logging.error(f"Validation error: {str(e)}")
        return jsonify({
            "error": str(e),
            "code": 400,
            "details": {"place": place, "bbox": bbox_str, "date": date}
        }), 400
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "code": 500,
            "details": str(e)
        }), 500


@app.route("/predict-gpkg", methods=["GET"])
def predict_gpkg():
    """Get pedestrian volume predictions as downloadable GPKG file - predictions only.
    
    Query Parameters:
        place (str): Place name (e.g., "Monaco", "Tel Aviv") - required
        bbox (str, optional): Bounding box as "minx,miny,maxx,maxy"
        date (str, optional): ISO timestamp (defaults to current time)
        
    Returns:
        GPKG file download with single 'predictions' layer and embedded QGIS styling
    """
    place = request.args.get("place")
    bbox_str = request.args.get("bbox")
    date = request.args.get("date")
    
    if not place:
        return jsonify({"error": "Missing 'place' parameter"}), 400

    try:
        # Validate and parse parameters
        place, bbox, date = validate_request_params(place, bbox_str, date)
        
        logging.info(f"GPKG prediction request for place={place}, bbox={bbox}, date={date}")
        
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not available",
                "code": 503,
                "details": "CatBoost model failed to load at startup"
            }), 503
        
        # Run feature extraction pipeline
        features_gdf, pipeline_metadata = run_feature_pipeline(
            place=place,
            bbox=bbox,
            timestamp=date
        )
        
        # Prepare features for model
        model_features = prepare_model_features(features_gdf)
        
        # Make predictions
        logging.info(f"Making predictions for {len(model_features)} edges")
        
        # Get categorical feature indices
        cat_feature_indices = [model_features.columns.get_loc(col) for col in CAT_COLS if col in model_features.columns]
        
        # Ensure categorical features are strings
        for col in CAT_COLS:
            if col in model_features.columns:
                model_features[col] = model_features[col].astype(str)
        
        # Create CatBoost Pool and predict
        pool = Pool(model_features, cat_features=cat_feature_indices)
        predictions = model.predict(pool)
        probabilities = model.predict_proba(pool)
        
        # Create predictions-only GeoDataFrame with proper field structure
        preds_gdf = features_gdf.copy()
        preds_gdf['volume_class'] = predictions.astype(int)
        
        # Add probability columns in the expected format (proba_1, proba_2, etc.)
        if probabilities.ndim > 1:
            for i in range(probabilities.shape[1]):
                preds_gdf[f'proba_{i+1}'] = probabilities[:, i]
            preds_gdf['proba_top'] = probabilities.max(axis=1)
        else:
            preds_gdf['proba_1'] = probabilities
            preds_gdf['proba_top'] = probabilities
        
        # Add model metadata if available
        try:
            preds_gdf["model_name"] = CURRENT_MODEL_NAME if 'CURRENT_MODEL_NAME' in globals() else "catboost_model"
            if 'MODEL_PATH' in globals():
                import hashlib
                with open(MODEL_PATH, 'rb') as f:
                    model_sha = hashlib.sha256(f.read()).hexdigest()[:16]
                preds_gdf["model_sha"] = model_sha
        except Exception:
            pass
        
        # Ensure EPSG:4326 CRS
        if preds_gdf.crs is None or str(preds_gdf.crs).upper() != "EPSG:4326":
            try:
                preds_gdf = preds_gdf.to_crs("EPSG:4326")
            except Exception:
                pass
        
        # Fix invalid geometries for QGIS
        try:
            preds_gdf.geometry = preds_gdf.geometry.apply(make_valid)
        except Exception:
            preds_gdf.geometry = preds_gdf.buffer(0)
        
        # Drop rows without geometry
        preds_gdf = preds_gdf[~preds_gdf.geometry.isna()].copy()
        if preds_gdf.empty:
            return jsonify({"error": "No features/geometry to export"}), 400

        # Windows-safe temp file
        fd, tmp_path = tempfile.mkstemp(suffix=".gpkg")
        os.close(fd)

        try:
            # Write single predictions layer
            preds_gdf.to_file(tmp_path, driver="GPKG", layer="predictions")
        except Exception:
            try:
                from pyogrio import write_dataframe
                write_dataframe(preds_gdf, tmp_path, driver="GPKG", layer="predictions")
            except Exception as e:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return jsonify({"error": f"Failed to write GPKG: {str(e)}"}), 500

        # Embed QGIS style (predictions only)
        try:
            _install_qgis_style(tmp_path, "predictions", _qml_predictions_categorized(),
                                style_name="predictions_default", use_default=1)
        except Exception as e:
            logging.warning(f"Could not install predictions style: {e}")

        # Generate filename
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"predictions_{_slug(place)}_{ts}.gpkg"

        @after_this_request
        def _cleanup(resp):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return resp

        logging.info(f"GPKG file created with {len(preds_gdf)} predictions for {place}")

        return send_file(
            tmp_path,
            mimetype="application/geopackage+sqlite3",
            as_attachment=True,
            download_name=fname,
            max_age=0,
            conditional=True,
        )

    except Exception as e:
        logging.exception("predict-gpkg failed")
        return jsonify({"error": str(e)}), 500


@app.route("/predict-batch-gpkg", methods=["POST"])
def predict_batch_gpkg():
    """Batch prediction endpoint that returns GPKG file.
    
    Request Body:
        {
            "items": [
                {
                    "edge_id": 1,
                    "betweenness": 0.3,
                    "closeness": 0.1,
                    "Hour": 8,
                    "is_weekend": 0,
                    "time_of_day": "morning",
                    "land_use": "retail",
                    "highway": "primary"
                },
                ...
            ],
            "place": "Monaco"  # Optional, for better geometries
        }
        
    Returns:
        GPKG file download with batch predictions and geometries
    """
    try:
        payload = request.get_json(silent=True) or {}
        items = payload.get("items", [])
        place = payload.get("place")
        
        if not items:
            return jsonify({"error": "Provide JSON with 'items': [...]"}), 400
        
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        logging.info(f"GPKG batch prediction request for {len(items)} items, place={place}")
        
        # Get predictions using existing function
        predictions = predict_with_model(items)
        
        # Build GeoDataFrame with geometries
        gdf = _build_predictions_gdf(predictions, place)
        
        if gdf.empty:
            return jsonify({"error": "No valid predictions generated"}), 400
        
        # Create temporary GPKG file
        import os
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.gpkg')
        
        try:
            # Close the file descriptor to allow geopandas to write
            os.close(tmp_fd)
            
            # Write the GPKG file
            gdf.to_file(tmp_path, driver='GPKG')
            
            # Generate filename
            place_slug = _slug(place) if place else "batch"
            filename = f"pedestrian_predictions_batch_{place_slug}.gpkg"
            
            logging.info(f"GPKG batch file created with {len(gdf)} predictions")
            
            # Send the file
            return send_file(
                tmp_path,
                as_attachment=True,
                download_name=filename,
                mimetype='application/geopackage+sqlite3'
            )
            
        except Exception as e:
            # Clean up on error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise e
            
    except Exception as e:
        logging.error(f"Batch GPKG prediction error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    # Development server; use Gunicorn in production
    import os
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "8000"))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    
    logging.info(f"Starting Flask development server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
