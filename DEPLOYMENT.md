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

### Model Retraining Schedule

```bash
# cron job for weekly retraining
# Add to crontab: crontab -e
0 2 * * 0 cd /path/to/crime_hotspot_optimizer && python main.py --city Chicago --mode predict --epochs 100
```

### Logging & Monitoring

```python
# monitoring/logger.py
import logging
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)
```

## Performance Optimization

### Model Caching
```python
import pickle
import hashlib

def cache_predictions(city, date):
    """Cache predictions to avoid recomputation"""
    cache_key = hashlib.md5(f"{city}_{date}".encode()).hexdigest()
    cache_file = f"cache/{cache_key}.pkl"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Generate predictions
    predictions = pipeline.predict_hotspots()
    
    # Cache result
    with open(cache_file, 'wb') as f:
        pickle.dump(predictions, f)
    
    return predictions
```

### Batch Processing
```python
def batch_predict_multiple_cities():
    """Generate predictions for all cities in batch"""
    results = {}
    for city in ['Chicago', 'LA']:
        pipeline = CrimeHotspotPipeline(city)
        pipeline.load_and_preprocess_data()
        pipeline.engineer_features()
        results[city] = pipeline.predict_hotspots()
    return results
```

## Security Considerations

### Authentication (if needed)
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(credentials = Depends(security)):
    token = credentials.credentials
    # Verify token against whitelist
    if token not in AUTHORIZED_TOKENS:
        raise HTTPException(status_code=401)
    return token

@app.get("/api/hotspots")
async def get_hotspots(token = Depends(verify_token)):
    # Endpoint now requires authentication
    return {...}
```

### Data Privacy
- Encrypt sensitive data at rest
- Use HTTPS for all communications
- Implement audit logging
- Regular security updates
- Data retention policies

## Backup & Recovery

### Database Backup
```bash
# Daily backup
0 3 * * * tar -czf /backups/predictions_$(date +\%Y\%m\%d).tar.gz /app/outputs
```

### Model Versioning
```python
def save_model_version(model, version):
    path = f"models/v{version}/model.pt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
```

## Scaling Strategy

### Horizontal Scaling
- Load balance multiple instances
- Use shared model cache
- Distribute batch predictions

### Vertical Scaling
- Increase server resources
- Use GPU for inference
- Optimize code performance

### Distributed Computing
```python
# Using Ray for distributed computation
import ray

@ray.remote
def predict_for_city(city):
    pipeline = CrimeHotspotPipeline(city)
    return pipeline.predict_hotspots()

# Run in parallel
results = ray.get([
    predict_for_city.remote('Chicago'),
    predict_for_city.remote('LA')
])
```

## Compliance & Regulations

### Data Protection
- GDPR compliance for European deployment
- California Consumer Privacy Act (CCPA)
- Local law enforcement data regulations

### Bias & Fairness
- Monitor for geographic bias
- Ensure fair resource allocation
- Regular fairness audits
- Community oversight

### Legal Requirements
- Privacy impact assessments
- Data retention policies
- Audit trails for all predictions
- Regular compliance reviews

---

**Ready for deployment!** Choose your deployment method and follow the setup instructions.
