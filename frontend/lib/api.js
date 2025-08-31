/**
 * API Client for Pedestrian Volume Prediction
 * Centralized API calls with environment-based configuration
 * Global functions for vanilla JS (non-module) usage
 */

// Get base URL from environment variables with fallback
function getBaseURL() {
    // For now, just return the default for local development
    // TODO: Add proper environment variable detection for different deployment scenarios
    return "http://127.0.0.1:8000";
}

const API_BASE_URL = getBaseURL();

// Make API_BASE_URL globally accessible
window.API_BASE_URL = API_BASE_URL;

/**
 * Generic API request helper
 * @param {string} endpoint - API endpoint path
 * @param {Object} options - Fetch options
 * @returns {Promise<Object>} - JSON response
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const defaultOptions = {
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        },
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(url, mergedOptions);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('API Error:', errorText);
            
            // Map HTTP status codes to user-friendly messages
            const errorMessages = {
                404: 'העיר לא נמצאה. אנא בדוק את השם ונסה שוב',
                500: 'שגיאה בשרת. אנא נסה שוב מאוחר יותר',
                503: 'השירות אינו זמין כרגע'
            };
            
            const message = errorMessages[response.status] || 
                           `שגיאת HTTP ${response.status}: ${response.statusText}`;
            throw new Error(message);
        }
        
        return await response.json();
        
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error(`לא ניתן להתחבר לשרת. ודא שהשרת רץ על ${API_BASE_URL}`);
        }
        throw error;
    }
}

/**
 * GET request helper
 * @param {string} endpoint - API endpoint
 * @param {Object} params - Query parameters
 * @returns {Promise<Object>} - JSON response
 */
async function apiGet(endpoint, params = {}) {
    const url = new URL(endpoint, API_BASE_URL);
    Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
            url.searchParams.set(key, value);
        }
    });
    
    return apiRequest(url.pathname + url.search, { method: 'GET' });
}

/**
 * POST request helper
 * @param {string} endpoint - API endpoint
 * @param {Object} data - Request body data
 * @returns {Promise<Object>} - JSON response
 */
async function apiPost(endpoint, data = {}) {
    return apiRequest(endpoint, {
        method: 'POST',
        body: JSON.stringify(data)
    });
}

/**
 * Health check endpoint
 * @returns {Promise<Object>} - Health status
 */
async function checkHealth() {
    return apiGet('/health');
}

/**
 * Get predictions for a place
 * @param {string} place - Place name (e.g., "Monaco", "Tel Aviv")
 * @param {string} bbox - Optional bounding box as "minx,miny,maxx,maxy"
 * @param {string} date - Optional ISO timestamp
 * @returns {Promise<Object>} - Prediction response with geojson
 */
async function getPredictions(place, bbox = null, date = null) {
    const params = { place };
    if (bbox) {
        params.bbox = bbox;
    }
    if (date) {
        params.date = date;
    }
    
    return apiGet('/predict', params);
}

/**
 * Get base network for a place or bbox
 * @param {string} place - Place name
 * @param {string} bbox - Bounding box as "w,s,e,n"
 * @param {number} maxFeatures - Maximum number of features
 * @returns {Promise<Object>} - GeoJSON network data
 */
async function getBaseNetwork(place = null, bbox = null, maxFeatures = 5000) {
    const params = { max_features: maxFeatures };
    if (place) params.place = place;
    if (bbox) params.bbox = bbox;
    
    return apiGet('/base-network', params);
}

/**
 * Simulate pedestrian volume with network edits
 * @param {string} place - Place name
 * @param {Array} edits - Array of edit operations
 * @param {string} bbox - Optional bounding box
 * @param {number} maxFeatures - Maximum number of features
 * @returns {Promise<Object>} - Simulation results with before/after comparisons
 */
async function simulateVolume(place, edits = [], bbox = null, maxFeatures = 8000) {
    const data = {
        place,
        edits,
        max_features: maxFeatures
    };
    
    if (bbox) {
        data.bbox = bbox;
    }
    
    return apiPost('/simulate', data);
}

/**
 * Get predictions as GPKG file download
 * @param {string} place - Place name (e.g., "Monaco", "Tel Aviv")
 * @param {string} date - Optional ISO timestamp
 * @returns {Promise<Blob>} - GPKG file blob for download
 */
async function getPredictionsGpkg(place, date = null) {
    const params = { place };
    if (date) {
        params.date = date;
    }
    
    const url = new URL('/predict-gpkg', API_BASE_URL);
    Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
            url.searchParams.set(key, value);
        }
    });
    
    const response = await fetch(url, {
        headers: {
            'Accept': 'application/geopackage+sqlite3',
        }
    });
    
    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response;
}

/**
 * Batch prediction as GPKG file download
 * @param {Array} items - Array of prediction items
 * @param {string} place - Optional place name for better geometries
 * @returns {Promise<Response>} - GPKG file response for download
 */
async function getPredictionsBatchGpkg(items, place = null) {
    const data = { items };
    if (place) {
        data.place = place;
    }
    
    const response = await fetch(`${API_BASE_URL}/predict-batch-gpkg`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/geopackage+sqlite3'
        },
        body: JSON.stringify(data)
    });
    
    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response;
}


// Make functions globally accessible
window.checkHealth = checkHealth;
window.getPredictions = getPredictions;
window.getBaseNetwork = getBaseNetwork;
window.simulateVolume = simulateVolume;
window.getPredictionsGpkg = getPredictionsGpkg;
window.getPredictionsBatchGpkg = getPredictionsBatchGpkg;
