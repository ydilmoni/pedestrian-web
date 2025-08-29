// script.js

class PedestrianPredictionApp {
    constructor() {
        this.API_BASE_URL = 'http://127.0.0.1:5000';
        this.map = null;
        this.currentLayer = null;
        this.currentBBoxRectangle = null;
        this.isDrawingBBox = false;
        
        this.initializeElements();
        this.initializeEventListeners();
        this.initializeMap();
    }
    
    initializeElements() {
        this.searchForm = document.getElementById('searchForm');
        this.cityInput = document.getElementById('cityInput');
        this.searchBtn = document.getElementById('searchBtn');
        this.buttonText = document.getElementById('buttonText');
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.statusMessage = document.getElementById('statusMessage');
        this.mapContainer = document.getElementById('mapContainer');
        this.mapTitle = document.getElementById('mapTitle');
        this.mapStats = document.getElementById('mapStats');
        this.loadingMessage = document.getElementById('loadingMessage');
        this.detailsContainer = document.getElementById('detailsContainer');
        this.predictionDetails = document.getElementById('predictionDetails');
        
        // Tools sidebar elements
        this.locateCityInput = document.getElementById('locateCityInput');
        this.locateCityBtn = document.getElementById('locateCityBtn');
        this.drawBBoxBtn = document.getElementById('drawBBoxBtn');
        this.clearBBoxBtn = document.getElementById('clearBBoxBtn');
    }
    
    initializeEventListeners() {
        // Main search form
        this.searchForm.addEventListener('submit', (e) => this.handleSearch(e));
        
        // Tools sidebar
        this.locateCityBtn.addEventListener('click', () => this.locateCity());
        this.drawBBoxBtn.addEventListener('click', () => this.startBBoxDraw());
        this.clearBBoxBtn.addEventListener('click', () => this.clearBBox());
        
        // Allow Enter key in locate city input
        this.locateCityInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.locateCity();
            }
        });
        
        // Clear error messages when user starts typing in main search
        this.cityInput.addEventListener('input', () => {
            if (!this.statusMessage.classList.contains('hidden') && 
                this.statusMessage.classList.contains('error')) {
                this.hideStatusMessage();
            }
        });
        
        // Allow Enter key in main city input
        this.cityInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.handleSearch(e);
            }
        });
    }
    
    initializeMap() {
        // Create map with default view (Israel as default)
        this.map = L.map('map').setView([31.5, 35.0], 8);
        
        console.log('Map initialized at:', this.map.getCenter(), 'zoom:', this.map.getZoom());
        
        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(this.map);
        
        // Add event listener to track when map view changes
        this.map.on('moveend', () => {
            console.log('Map moved to:', this.map.getCenter(), 'zoom:', this.map.getZoom());
        });
    }
    
    // Locate city function - only moves map, no API call
    async locateCity(cityName) {
        const city = cityName || this.locateCityInput.value.trim();
        if (!city) {
            this.showStatusMessage('אנא הזן שם עיר', 'error');
            return;
        }
        
        console.log('locateCity called with:', city);
        
        // DON'T locate if this is a bbox format
        if (this.isBBoxFormat(city)) {
            console.log('Input is BBOX format, skipping location');
            return;
        }
        
        try {
            const geocodeUrl = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(city)}&limit=1`;
            
            const response = await fetch(geocodeUrl);
            const results = await response.json();
            
            if (results && results.length > 0) {
                const location = results[0];
                const lat = parseFloat(location.lat);
                const lon = parseFloat(location.lon);
                
                console.log('Moving map to:', lat, lon);
                this.map.setView([lat, lon], 12, { animate: true });
                console.log(`Moved to ${city}: ${lat}, ${lon}`);
                
                // Clear the locate input after successful location
                if (!cityName) { // Only clear if called from sidebar, not main search
                    this.locateCityInput.value = '';
                }
            } else {
                this.showStatusMessage(`לא נמצא מיקום עבור: ${city}`, 'error');
            }
        } catch (error) {
            console.error('Error finding location:', error);
            this.showStatusMessage('שגיאה בחיפוש מיקום', 'error');
        }
    }
    
    // Start BBox drawing
    startBBoxDraw() {
        if (this.isDrawingBBox) return;
        
        this.isDrawingBBox = true;
        this.drawBBoxBtn.disabled = true;
        this.drawBBoxBtn.textContent = 'מסמן...';
        
        // Clear any existing bbox rectangle
        if (this.currentBBoxRectangle) {
            this.map.removeLayer(this.currentBBoxRectangle);
            this.currentBBoxRectangle = null;
        }
        
        // Disable map interactions
        this.map.dragging.disable();
        this.map.boxZoom.disable();
        this.map.doubleClickZoom.disable();
        
        // Change cursor
        this.map.getContainer().style.cursor = 'crosshair';
        
        // Set up drawing
        let startPoint = null;
        let tempRectangle = null;
        
        const onMouseDown = (e) => {
            startPoint = e.latlng;
            
            const onMouseMove = (moveEvent) => {
                if (!startPoint) return;
                
                const currentPoint = moveEvent.latlng;
                
                // Remove previous temp rectangle
                if (tempRectangle) {
                    this.map.removeLayer(tempRectangle);
                }
                
                // Create bounds for rectangle
                const bounds = L.latLngBounds(startPoint, currentPoint);
                
                // Draw temporary rectangle
                tempRectangle = L.rectangle(bounds, {
                    color: '#3498db',
                    fillColor: '#3498db',
                    fillOpacity: 0.2,
                    weight: 2,
                    dashArray: '5, 5'
                }).addTo(this.map);
            };
            
            const onMouseUp = (upEvent) => {
                // Remove temp rectangle
                if (tempRectangle) {
                    this.map.removeLayer(tempRectangle);
                }
                
                // Create final rectangle
                const endPoint = upEvent.latlng;
                const bounds = L.latLngBounds(startPoint, endPoint);
                
                this.currentBBoxRectangle = L.rectangle(bounds, {
                    color: '#e74c3c',
                    fillColor: '#e74c3c',
                    fillOpacity: 0.1,
                    weight: 3
                }).addTo(this.map);
                
                // Extract bbox coordinates in correct order: west,south,east,north
                const sw = bounds.getSouthWest();
                const ne = bounds.getNorthEast();
                const bboxString = `${sw.lng.toFixed(6)},${sw.lat.toFixed(6)},${ne.lng.toFixed(6)},${ne.lat.toFixed(6)}`;
                
                // Inject bbox string into main search input
                this.cityInput.value = bboxString;
                
                // Clean up
                this.finishBBoxDraw();
                
                // Remove event listeners
                this.map.off('mousemove', onMouseMove);
                this.map.off('mouseup', onMouseUp);
            };
            
            // Add temporary event listeners
            this.map.on('mousemove', onMouseMove);
            this.map.on('mouseup', onMouseUp);
        };
        
        // Add mousedown listener
        this.map.once('mousedown', onMouseDown);
    }
    
    finishBBoxDraw() {
        this.isDrawingBBox = false;
        this.drawBBoxBtn.disabled = false;
        this.drawBBoxBtn.textContent = 'התחל סימון';
        
        // Re-enable map interactions
        this.map.dragging.enable();
        this.map.boxZoom.enable();
        this.map.doubleClickZoom.enable();
        
        // Reset cursor
        this.map.getContainer().style.cursor = '';
    }
    
    // Clear BBox rectangle
    clearBBox() {
        if (this.currentBBoxRectangle) {
            this.map.removeLayer(this.currentBBoxRectangle);
            this.currentBBoxRectangle = null;
        }
        
        // Clear the main search input if it contains bbox format
        const currentInput = this.cityInput.value.trim();
        if (this.isBBoxFormat(currentInput)) {
            this.cityInput.value = '';
        }
    }
    
    // Check if input is bbox format
    isBBoxFormat(input) {
        const bboxRegex = /^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$/;
        return bboxRegex.test(input);
    }
    
    // Main search handler - decides between place and bbox
    async handleSearch(event) {
        event.preventDefault();
        
        const input = this.cityInput.value.trim();
        console.log('handleSearch called with input:', input);
        
        if (!input) {
            this.showStatusMessage('אנא הזן שם עיר או bbox', 'error');
            return;
        }
        
        // Clear any existing error messages
        this.hideStatusMessage();
        
        this.setLoading(true);
        this.showLoadingOnMap(true);
        
        const isBBox = this.isBBoxFormat(input);
        console.log('Is BBOX format:', isBBox);
        
        try {
            let predictions;
            
            if (isBBox) {
                console.log('Processing as BBOX');
                // Search by bbox - DON'T move to location, keep current view
                predictions = await this.fetchPredictionsByBBox(input);
                console.log('BBOX predictions received:', predictions);
                this.displayResults(predictions, 'איזור נבחר');
            } else {
                console.log('Processing as place name');
                // Search by place - move to location first
                await this.locateCity(input);
                predictions = await this.fetchPredictionsByPlace(input);
                console.log('Place predictions received:', predictions);
                this.displayResults(predictions, input);
            }
        } catch (error) {
            console.error('Prediction error:', error);
            this.showStatusMessage(`שגיאה: ${error.message}`, 'error');
        } finally {
            this.setLoading(false);
            this.showLoadingOnMap(false);
        }
    }
    
    // Fetch predictions by place
    async fetchPredictionsByPlace(place) {
        const params = new URLSearchParams({
            place: place
        });
        
        return this.fetchPredictions(params.toString());
    }
    
    // Fetch predictions by bbox
    async fetchPredictionsByBBox(bbox) {
        const params = new URLSearchParams({
            bbox: bbox
        });
        
        return this.fetchPredictions(params.toString());
    }
    
    // Common fetch function
    async fetchPredictions(queryString) {
        try {
            const response = await fetch(`${this.API_BASE_URL}/predict?${queryString}`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                },
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Response error:', errorText);
                
                if (response.status === 404) {
                    throw new Error('לא נמצאו נתונים. אנא בדוק את הפרמטרים ונסה שוב');
                } else if (response.status === 500) {
                    throw new Error('שגיאה בשרת. אנא נסה שוב מאוחר יותר');
                } else if (response.status === 503) {
                    throw new Error('השירות אינו זמין כרגע');
                } else {
                    throw new Error(`שגיאת HTTP ${response.status}: ${response.statusText}`);
                }
            }
            
            const data = await response.json();
            console.log('Received data:', data);
            
            if (!data.geojson || !data.geojson.features) {
                throw new Error('לא התקבלו נתוני מפה מהשרת');
            }
            
            return data;
            
        } catch (error) {
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('לא ניתן להתחבר לשרת. ודא שהשרת רץ על http://127.0.0.1:5000');
            }
            throw error;
        }
    }
    
    showLoadingOnMap(show) {
        if (show) {
            this.loadingMessage.classList.remove('hidden');
            // DON'T reset map title or stats here - keep current state
        } else {
            this.loadingMessage.classList.add('hidden');
        }
    }
    
    displayResults(data, locationName) {
        console.log('displayResults called with locationName:', locationName);
        console.log('Data received:', data);
        
        // Show details container
        this.detailsContainer.classList.remove('hidden');
        
        // Extract stats from response
        const numFeatures = data.geojson && data.geojson.features ? data.geojson.features.length : 0;
        const processingTime = data.processing_time || 'לא זמין';
        const networkStats = data.network_stats || { n_edges: numFeatures, n_nodes: 0 };
        
        console.log('Number of features to display:', numFeatures);
        
        // Update title and stats
        this.mapTitle.textContent = `תחזית נפח הולכי רגל - ${locationName}`;
        this.mapStats.innerHTML = `
            <span>מספר רחובות: ${networkStats.n_edges}</span>
            <span>זמן עיבוד: ${processingTime}s</span>
        `;
        
        // Update map with results - but preserve current view for bbox
        const preserveView = locationName === 'איזור נבחר';
        console.log('Calling updateMapWithData with preserveView:', preserveView);
        this.updateMapWithData(data.geojson, preserveView);
        
        // Create sample prediction from first feature if not provided
        let samplePrediction = data.sample_prediction;
        if (!samplePrediction && data.geojson.features && data.geojson.features.length > 0) {
            const firstFeature = data.geojson.features[0];
            samplePrediction = {
                volume_bin: firstFeature.properties.volume_bin,
                features: {
                    Hour: firstFeature.properties.Hour,
                    is_weekend: firstFeature.properties.is_weekend,
                    time_of_day: firstFeature.properties.time_of_day,
                    highway: firstFeature.properties.highway,
                    land_use: firstFeature.properties.land_use
                }
            };
        }
        
        // Update details
        this.updatePredictionDetails({
            sample_prediction: samplePrediction,
            network_stats: networkStats,
            processing_time: processingTime,
            validation: data.validation || { warnings: [] }
        });
        
        // Scroll to results
        this.mapContainer.scrollIntoView({ behavior: 'smooth' });
        
        console.log('displayResults completed');
    }
    
    updateMapWithData(geojson, preserveView = false) {
        console.log('updateMapWithData called with preserveView:', preserveView);
        console.log('GeoJSON features count:', geojson.features ? geojson.features.length : 0);
        console.log('Current map center before update:', this.map.getCenter());
        console.log('Current map zoom before update:', this.map.getZoom());
        
        // Remove previous layer
        if (this.currentLayer) {
            this.map.removeLayer(this.currentLayer);
            console.log('Removed previous layer');
        }
        
        // Add GeoJSON layer
        this.currentLayer = L.geoJSON(geojson, {
            style: (feature) => this.getFeatureStyle(feature),
            onEachFeature: (feature, layer) => this.bindFeaturePopup(feature, layer)
        }).addTo(this.map);
        
        console.log('New layer added to map');
        console.log('Layer bounds:', this.currentLayer.getBounds());
        
        // Only fit bounds if NOT preserving view (i.e., for city search, not bbox)
        if (!preserveView && this.currentLayer.getBounds().isValid()) {
            console.log('Fitting bounds to new layer');
            this.map.fitBounds(this.currentLayer.getBounds(), { padding: [20, 20] });
        } else {
            console.log('Preserving current view - not fitting bounds');
        }
        
        console.log('Current map center after update:', this.map.getCenter());
        console.log('Current map zoom after update:', this.map.getZoom());
    }
    
    getFeatureStyle(feature) {
        const volumeBin = feature.properties.volume_bin || 1;
        
        // Color scheme based on volume bin - updated colors
        const colors = {
            1: '#28c20aff',
            2: '#35bdb4ff', 
            3: '#3628b5ff',
            4: '#e53bd1ff',
            5: '#b00808ff'
        };
        
        // Width based on volume bin
        const widths = {
            1: 2,
            2: 2.5,
            3: 3,
            4: 3.5,
            5: 4
        };
        
        return {
            color: colors[volumeBin] || colors[1],
            weight: widths[volumeBin] || widths[1],
            opacity: 0.8
        };
    }
    
    bindFeaturePopup(feature, layer) {
        const props = feature.properties;
        
        const popupContent = `
            <div style="direction: rtl; text-align: right;">
                <h4>פרטי רחוב</h4>
                <p><strong>נפח חזוי:</strong> ${props.volume_bin || 'N/A'}</p>
                <p><strong>סוג רחוב:</strong> ${this.translateHighway(props.highway) || 'לא ידוע'}</p>
                <p><strong>שימוש קרקע:</strong> ${this.translateLandUse(props.land_use) || 'לא ידוע'}</p>
                <p><strong>אורך:</strong> ${Math.round(props.length || 0)} מטר</p>
                ${props.osmid ? `<p><strong>מזהה OSM:</strong> ${props.osmid}</p>` : ''}
            </div>
        `;
        
        layer.bindPopup(popupContent);
    }
    
    translateHighway(highway) {
        const translations = {
            'primary': 'כביש ראשי',
            'secondary': 'כביש משני', 
            'tertiary': 'כביש שלישוני',
            'residential': 'רחוב מגורים',
            'footway': 'שביל הולכי רגל',
            'path': 'שביל',
            'pedestrian': 'אזור הולכי רגל',
            'living_street': 'רחוב מגורים שקט',
            'unclassified': 'לא מסווג',
            'service': 'דרך שירות'
        };
        return translations[highway] || highway;
    }
    
    translateLandUse(landUse) {
        const translations = {
            'residential': 'מגורים',
            'commercial': 'מסחרי',
            'retail': 'קמעונאי',
            'industrial': 'תעשייתי',
            'other': 'אחר'
        };
        return translations[landUse] || landUse;
    }
    
    updatePredictionDetails(data) {
        const sample = data.sample_prediction;
        const features = sample?.features || {};
        const stats = data.network_stats;
        
        this.predictionDetails.innerHTML = `
            <div class="detail-item">
                <div class="detail-label">נפח חזוי לדוגמה</div>
                <div class="detail-value highlight">${sample?.volume_bin || 'N/A'}</div>
            </div>
            
            <div class="detail-item">
                <div class="detail-label">שעה</div>
                <div class="detail-value">${features.Hour || 'N/A'}</div>
            </div>
            
            <div class="detail-item">
                <div class="detail-label">סוף שבוע</div>
                <div class="detail-value">${features.is_weekend ? 'כן' : 'לא'}</div>
            </div>
            
            <div class="detail-item">
                <div class="detail-label">זמן ביום</div>
                <div class="detail-value">${this.translateTimeOfDay(features.time_of_day)}</div>
            </div>
            
            <div class="detail-item">
                <div class="detail-label">מספר רחובות</div>
                <div class="detail-value">${stats.n_edges || 0}</div>
            </div>
            
            <div class="detail-item">
                <div class="detail-label">מספר צמתים</div>
                <div class="detail-value">${stats.n_nodes || 0}</div>
            </div>
            
            <div class="detail-item">
                <div class="detail-label">זמן עיבוד</div>
                <div class="detail-value">${data.processing_time}s</div>
            </div>
            
            ${data.validation?.warnings?.length > 0 ? `
                <div class="detail-item">
                    <div class="detail-label">אזהרות</div>
                    <div class="detail-value">${data.validation.warnings.join(', ')}</div>
                </div>
            ` : ''}
        `;
    }
    
    translateTimeOfDay(timeOfDay) {
        const translations = {
            'morning': 'בוקר',
            'afternoon': 'אחר צהריים', 
            'evening': 'ערב',
            'night': 'לילה'
        };
        return translations[timeOfDay] || timeOfDay;
    }
    
    setLoading(loading) {
        this.searchBtn.disabled = loading;
        this.cityInput.disabled = loading;
        
        if (loading) {
            this.buttonText.classList.add('hidden');
            this.loadingSpinner.classList.remove('hidden');
        } else {
            this.buttonText.classList.remove('hidden');
            this.loadingSpinner.classList.add('hidden');
        }
    }
    
    showStatusMessage(message, type = 'info') {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status-message ${type}`;
        this.statusMessage.classList.remove('hidden');
        
        // Only auto-hide success messages, keep error messages until user action
        if (type === 'success') {
            setTimeout(() => {
                this.hideStatusMessage();
            }, 3000);
        }
    }
    
    hideStatusMessage() {
        this.statusMessage.classList.add('hidden');
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PedestrianPredictionApp();
});