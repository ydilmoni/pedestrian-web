# Pedestrian Volume Prediction Web Application

A standalone web application for predicting pedestrian volume on street networks using machine learning models.

## Project Structure

```
pedestrian-web/
├── api/                           # Backend API
│   ├── app.py                     # Flask API server
│   ├── requirements.txt           # Python dependencies
│   ├── models/                    # CatBoost ML models (.cbm files, Git LFS tracked)
│   ├── feature_engineering/       # Feature extraction modules
│   │   ├── centrality_features.py
│   │   ├── highway_features.py
│   │   ├── landuse_features.py
│   │   └── time_features.py
│   ├── cache/                     # API response cache
│   └── temp/                      # Temporary GeoPackage files
├── frontend/                      # Frontend web interface
│   ├── index.html                 # Main HTML page
│   ├── script.js                  # JavaScript application logic
│   ├── styles.css                 # CSS styling
│   └── lib/
│       └── api.js                 # API client library
├── .gitattributes                 # Git LFS configuration
└── .gitignore                     # Git ignore rules
```

## Quick Start

### Backend (API Server)

1. Navigate to the API directory:
   ```bash
   cd api/
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask development server:
   ```bash
   python app.py
   ```
   
   The API will be available at `http://127.0.0.1:8000`

### Frontend (Web Interface)

1. Navigate to the frontend directory:
   ```bash
   cd frontend/
   ```

2. Start a simple HTTP server:
   ```bash
   # Using Python
   python -m http.server 3000
   
   # Or using Node.js
   npx http-server -p 3000
   ```

3. Open your browser and go to: `http://localhost:3000`

## Features

- **Street Network Analysis**: Fetches and analyzes OpenStreetMap data
- **ML-based Predictions**: Uses CatBoost models trained on multiple cities
- **Interactive Visualization**: Leaflet.js map with color-coded predictions
- **GPKG Export**: Download predictions as GeoPackage with embedded QGIS styling
- **Multi-language Support**: Hebrew/English interface

## API Endpoints

- `GET /health` - Health check
- `GET /predict?place={city}` - Get predictions for a city (JSON)
- `GET /predict-gpkg?place={city}` - Download predictions as GPKG
- `GET /base-network?place={city}` - Get base street network (JSON)
- `POST /simulate` - Simulate network modifications

## Git LFS Setup

This project uses Git LFS for machine learning model files (.cbm):

1. Install Git LFS:
   ```bash
   git lfs install
   ```

2. The `.gitattributes` file is already configured to track model files.

## Development

The application is designed as a standalone system with:
- Flask API backend serving ML predictions
- Vanilla JavaScript frontend (no build process required)
- Cached responses for improved performance
- Environment-based configuration support

## Dependencies

### Backend
- Flask, Flask-CORS
- CatBoost, NetworkX, OSMnx
- Geopandas, GDAL, Fiona
- Requests, Shapely

### Frontend  
- Leaflet.js for mapping
- Native fetch API for HTTP requests
- CSS Grid and Flexbox for layout

## Deploy on Render

This repo contains a `render.yaml` blueprint for:
- **Web service**: Flask API (`/api`) via Gunicorn
- **Static site**: frontend (`/frontend`)
- **Persistent Disk**: `/var/data` for generated GPKGs/caches
- **Git LFS**: models pulled at build time

### Steps
1. Push to GitHub.
2. On Render → **New** → **Blueprint** → select this repo → **Apply**.
3. Wait for the API and static site to deploy.
4. Copy the API URL (e.g., `https://pedestrian-api.onrender.com`) and open the frontend site.
5. (Optional) Set a permanent base:
   - Open the frontend, add `?api=<YOUR_API_URL>` once, e.g.:  
     `https://pedestrian-frontend.onrender.com/?api=https://pedestrian-api.onrender.com`  
     The app saves this in `localStorage` for next visits.

### Environment variables
- `CORS_ORIGINS` (comma-separated): set to your static site + localhost.
- `DATA_DIR`: default `/var/data` (disk mount).

### Notes
- Models (`api/models/*.cbm`) are tracked via Git LFS and pulled on build.
- GPKGs are created in `/var/data` and streamed back to the client.

## License

This project is part of pedestrian volume prediction research.
