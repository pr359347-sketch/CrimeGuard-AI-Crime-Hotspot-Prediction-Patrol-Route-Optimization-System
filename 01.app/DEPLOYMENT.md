# Deployment Guide

## Production Deployment Options

### Option 1: Local Deployment (Police Department)

**Requirements:**
- Windows/Linux/macOS server with Python 3.8+
- 8GB RAM minimum
- 10GB disk space

**Setup:**
```bash
# Clone/extract project
cd crime_hotspot_optimizer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure for your jurisdiction
vim src/config.py  # Edit city, grid size, patrol units

# Start dashboard
streamlit run streamlit_app.py --server.port 8501
```

**Access:** `http://server-ip:8501`

### Option 2: Docker Container Deployment

**Build Docker image:**
```bash
docker build -t crime-hotspot:latest .
```

**Run container:**
```bash
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  --name crime-hotspot \
  crime-hotspot:latest
```

**Deploy on Docker Compose:**
```yaml
version: '3.8'

services:
  crime-hotspot:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./outputs:/app/outputs
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

### Option 3: Cloud Deployment (AWS/Azure/GCP)

**AWS EC2:**
```bash
# Launch Ubuntu instance
# SSH into instance
ssh -i key.pem ubuntu@instance-ip

# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv

# Clone and deploy
git clone <repository>
cd crime_hotspot_optimizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
nohup streamlit run streamlit_app.py --server.port 8501 &
```

**Heroku Deployment:**
```bash
# Install Heroku CLI
brew install heroku/brew/heroku

# Create Heroku app
heroku create crime-hotspot-app

# Deploy
git push heroku main

# Scale dyos
heroku ps:scale web=1
```

**Google Cloud Run:**
```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/crime-hotspot

# Deploy
gcloud run deploy crime-hotspot \
  --image gcr.io/PROJECT_ID/crime-hotspot \
  --platform managed \
  --region us-central1
```

## Integration with Dispatch Systems

### API Endpoint Development

Create `api/server.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.insert(0, 'src')

from prediction_pipeline import CrimeHotspotPipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = CrimeHotspotPipeline('Chicago')

@app.get("/api/hotspots")
async def get_hotspots():
    """Get current hotspot predictions"""
    predictions = pipeline.predict_hotspots()
    hotspots = pipeline.detect_hotspots()
    return {
        "hotspots": hotspots,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/routes")
async def get_routes():
    """Get optimized patrol routes"""
    routes = pipeline.routes
    return {"routes": routes}

@app.post("/api/train")
async def trigger_training():
    """Trigger model retraining"""
    pipeline.train_model(epochs=50)
    return {"status": "training_complete"}

# Run: uvicorn api.server:app --reload --port 8000
```

### Integration with CAD (Computer-Aided Dispatch)

**Generic API Integration:**
```python
import requests
import json

def get_hotspots_from_prediction_system():
    """Fetch hotspots from deployed system"""
    response = requests.get('http://prediction-system:8000/api/hotspots')
    return response.json()

def update_dispatch_priorities(hotspots):
    """Update CAD priorities based on predictions"""
    cad_api = 'http://cad-system:5000/api/update-priorities'
    
    priorities = []
    for hotspot in hotspots:
        priorities.append({
            'zone': hotspot,
            'priority': 'HIGH',
            'recommended_units': 2
        })
    
    response = requests.post(cad_api, json=priorities)
    return response.status_code == 200

# Main integration loop
while True:
    hotspots = get_hotspots_from_prediction_system()
    update_dispatch_priorities(hotspots)
    time.sleep(3600)  # Update every hour
```

## Monitoring & Maintenance

### Health Checks

```python
# monitoring/health_check.py
import requests
from datetime import datetime

def check_system_health():
    """Monitor system health"""
    status = {
        'timestamp': datetime.now().isoformat(),
        'api_available': False,
        'model_loaded': False,
        'last_prediction': None
    }
    
    try:
        response = requests.get('http://localhost:8000/api/hotspots', timeout=5)
        status['api_available'] = response.status_code == 200
    except:
        pass
    
    return status
```

## ... (truncated)