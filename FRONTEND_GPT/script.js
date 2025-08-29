class PedestrianPredictionApp {
    constructor() {
        this.API_BASE_URL = 'http://127.0.0.1:5000';
        this.map = null;
        this.currentLayer = null;
        this.lastGeoJSON = null;
        this.lastCityName = '';
        
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
        this.downloadBtn = document.getElementById('downloadBtn');
        this.statusMessage = document.getElementById('statusMessage');
        this.mapContainer = document.getElementById('mapContainer');
        this.mapTitle = document.getElementById('mapTitle');
        this.mapStats = document.getElementById('mapStats');
        this.loadingMessage = document.getElementById('loadingMessage');
        this.detailsContainer = document.getElementById('detailsContainer');
        this.predictionDetails = document.getElementById('predictionDetails');
    }
    
    initializeEventListeners() {
        this.searchForm.addEventListener('submit', (e) => this.handleSearch(e));
        
        // Clear error messages when user starts typing
        this.cityInput.addEventListener('input', () => {
            if (!this.statusMessage.classList.contains('hidden') && 
                this.statusMessage.classList.contains('error')) {
                this.hideStatusMessage();
            }
        });
        
        // Allow Enter key in city input
        this.cityInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.handleSearch(e);
            }
        });
        
        this.downloadBtn.addEventListener('click', () => this.downloadGeoJSON());
    }
    
    initializeMap() {
        // Create map with default view (Israel as default)
        this.map = L.map('map').setView([31.5, 35.0], 8);
        
        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(this.map);
    }
    
    async handleSearch(event) {
        event.preventDefault();
        
        const city = this.cityInput.value.trim();
        if (!city) {
            this.showStatusMessage('אנא הזן שם עיר', 'error');
            return;
        }
        
        // Clear any existing error messages
        this.hideStatusMessage();
        
        this.setLoading(true);
        this.showLoadingOnMap(true);
        this.downloadBtn.disabled = true;
        
        // זוז למיקום מיד לפני הבקשה לשרת
        await this.searchAndMoveToLocation(city);
        
        try {
            const predictions = await this.fetchPredictions(city);
            this.displayResults(predictions, city);
        } catch (error) {
            console.error('Prediction error:', error);
            this.showStatusMessage(`שגיאה: ${error.message}`, 'error');
        } finally {
            this.setLoading(false);
            this.showLoadingOnMap(false);
        }
    }
    
    showLoadingOnMap(show) {
        if (show) {
            this.loadingMessage.classList.remove('hidden');
            this.mapTitle.textContent = 'מפת רחובות';
            this.mapStats.innerHTML = '';
        } else {
            this.loadingMessage.classList.add('hidden');
        }
    }
    
    async searchAndMoveToLocation(cityName) {
        try {
            // חיפוש מיקום באמצעות Nominatim
            const geocodeUrl = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(cityName)}&limit=1`;
            
            const response = await fetch(geocodeUrl);
            const results = await response.json();
            
            if (results && results.length > 0) {
                const location = results[0];
                const lat = parseFloat(location.lat);
                const lon = parseFloat(location.lon);
                
                // זוז למיקום מיד
                this.map.setView([lat, lon], 12, { animate: true });
                
                console.log(`Moved to ${cityName}: ${lat}, ${lon}`);
            } else {
                console.log(`Location not found for: ${cityName}`);
            }
        } catch (error) {
            console.error('Error finding location:', error);
            // אם החיפוש נכשל, לא נעשה כלום - המפה תישאר במיקום הנוכחי
        }
    }
    
    async fetchPredictions(city) {
        const params = new URLSearchParams({
            place: city
        });
        
        try {
            const response = await fetch(`${this.API_BASE_URL}/predict?${params.toString()}`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                },
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Response error:', errorText);
                
                if (response.status === 404) {
                    throw new Error('העיר לא נמצאה. אנא בדוק את השם ונסה שוב');
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
            
            // בדוק אם יש geojson בתשובה
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
    
    displayResults(data, cityName) {
        // Show details container
        this.detailsContainer.classList.remove('hidden');
        // Store last results for download
        this.lastGeoJSON = data.geojson;
        this.lastCityName = cityName;
        this.downloadBtn.disabled = false;
        
        // Extract stats from response
        const numFeatures = data.geojson && data.geojson.features ? data.geojson.features.length : 0;
        const processingTime = data.processing_time || 'לא זמין';
        const networkStats = data.network_stats || { n_edges: numFeatures, n_nodes: 0 };
        
        // Update title and stats
        this.mapTitle.textContent = `תחזית נפח הולכי רגל - ${cityName}`;
        this.mapStats.innerHTML = `
            <span>מספר רחובות: ${networkStats.n_edges}</span>
            <span>זמן עיבוד: ${processingTime}s</span>
        `;
        
        // Update map with results
        this.updateMapWithData(data.geojson);
        
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
    }
    
    updateMapWithData(geojson) {
        // Remove previous layer
        if (this.currentLayer) {
            this.map.removeLayer(this.currentLayer);
        }
        
        // Add GeoJSON layer
        this.currentLayer = L.geoJSON(geojson, {
            style: (feature) => this.getFeatureStyle(feature),
            onEachFeature: (feature, layer) => this.bindFeaturePopup(feature, layer)
        }).addTo(this.map);
        
        // Fit map to bounds
        if (this.currentLayer.getBounds().isValid()) {
            this.map.fitBounds(this.currentLayer.getBounds(), { padding: [20, 20] });
        }
    }
    
    getFeatureStyle(feature) {
        const volumeBin = feature.properties.volume_bin || 1;
        
        // Color scheme based on volume bin
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
    
    downloadGeoJSON() {
        if (!this.lastGeoJSON) return;
        try {
            const geojsonStr = JSON.stringify(this.lastGeoJSON, null, 2);
            const blob = new Blob([geojsonStr], { type: 'application/json' });
            const fileCityName = this.lastCityName.replace(/[^a-zA-Z0-9\u0590-\u05FF_]+/g, '_');
            const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '_');
            const filename = `${fileCityName}_${timestamp}.geojson`;
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Download failed:', error);
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PedestrianPredictionApp();
});
