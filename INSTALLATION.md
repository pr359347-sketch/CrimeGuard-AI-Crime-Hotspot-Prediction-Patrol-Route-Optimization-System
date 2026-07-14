# Installation & Setup Guide

## System Requirements

- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows
- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 5GB free space

## Step-by-Step Installation

### 1. Clone or Download the Project

```bash
cd ~/projects
# If using git:
# git clone <repository-url>
# Or extract the downloaded archive
cd crime_hotspot_optimizer
```

### 2. Create Virtual Environment

```bash
# Using venv
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Note**: First-time installation may take 5-10 minutes due to PyTorch and PyTorch Geometric compilation.

### 4. Verify Installation

```bash
# Test imports
python3 -c "import torch; import torch_geometric; import streamlit; print('✓ All dependencies installed')"

# Check data directory structure
mkdir -p data models outputs
```

## Running the System

### Option 1: Interactive Dashboard (Recommended)

```bash
# Start Streamlit app
streamlit run streamlit_app.py

# Opens automatically at http://localhost:8501
```

### Option 2: Command-Line Interface

```bash
# Full pipeline for Chicago
python main.py --city Chicago --mode full --epochs 10

# Output:
# - Predictions: outputs/chicago_heatmap.html
# - Routes: outputs/chicago_routes.html
# - Models: models/chicago_model.pt

# Only prediction (higher accuracy)
python main.py --city Chicago --mode predict --epochs 50

# Only optimization
python main.py --city Chicago --mode optimize

# For LA
python main.py --city LA --mode full
```

### Option 3: Python Script

```python
from src.prediction_pipeline import CrimeHotspotPipeline

# Initialize
pipeline = CrimeHotspotPipeline('Chicago')

# Run pipeline
pipeline.load_and_preprocess_data()
pipeline.engineer_features()
pipeline.train_model(epochs=20)
pipeline.predict_hotspots()
pipeline.detect_hotspots()
pipeline.optimize_patrol_routes()
pipeline.generate_visualizations()

print(f"Hotspots: {len(pipeline.hotspots)}")
print(f"Routes: {len(pipeline.routes)}")
```

## Configuration

### Edit `src/config.py` for custom settings:

```python
# Grid resolution (higher = finer predictions, slower)
GRID_SIZE = 50  # Try 30-50

# Historical data window
HISTORY_WINDOW = 60  # days (30-90 recommended)

# Prediction horizon
PREDICTION_HORIZON = 7  # days (1-14 range)

# Model capacity
ST_GNN_HIDDEN_DIM = 64  # 32-128 depending on GPU
LSTM_HIDDEN_DIM = 128  # 64-256 depending on GPU

# Training
EPOCHS = 50  # 10-100 depending on data size
LEARNING_RATE = 0.001  # 0.0001-0.01

# Patrol units
NUM_PATROL_UNITS = 10  # Your actual fleet size
```

## Troubleshooting

### Issue: CUDA/GPU not detected
```bash
# Use CPU instead
# Edit src/config.py:
# config.DEVICE = 'cpu'

# Or install CUDA-compatible PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Memory errors
```bash
# Reduce batch size in src/config.py:
BATCH_SIZE = 16  # was 32

# Or reduce grid size:
GRID_SIZE = 30  # was 50
```

### Issue: Data download fails
```bash
# The system automatically generates synthetic data
# No action needed - just run the pipeline
# Real data from APIs will be used if connection available
```

### Issue: Streamlit not opening
```bash
# Check if port 8501 is available
lsof -i :8501  # Linux/macOS

# Use alternative port
streamlit run streamlit_app.py --server.port 8502
```

### Issue: Module not found errors
```bash
# Ensure virtual environment is activated
# Re-install requirements
pip install -r requirements.txt --force-reinstall

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/crime_hotspot_optimizer"
```

## Performance Tips

### For Faster Results:
```python
# In main.py or scripts:
EPOCHS = 10  # Instead of 50
GRID_SIZE = 30  # Instead of 50
BATCH_SIZE = 64  # Instead of 32
```

### For Better Accuracy:
```python
EPOCHS = 100
GRID_SIZE = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
```

### For Production:
```python
# Use GPU
config.DEVICE = 'cuda'

# Increase model capacity
ST_GNN_HIDDEN_DIM = 128
LSTM_HIDDEN_DIM = 256

# More training data
HISTORY_WINDOW = 180  # 6 months

# Regular retraining
# Schedule: Weekly or after significant events
```

## Docker Setup (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.headless", "true"]
```

Build and run:
```bash
docker build -t crime-hotspot-optimizer .
docker run -p 8501:8501 crime-hotspot-optimizer
```

## Data Sources

### Chicago Crime Data
- **API**: https://data.cityofchicago.org/api/views/ijzp-q8t2
- **Format**: JSON via Socrata API
- **Records**: 50,000+ per run
- **Coverage**: Historical (several years)

### LA Crime Data
- **API**: https://data.lacity.gov/api/views/2nrs-mtv8
- **Format**: JSON via Socrata API
- **Records**: 50,000+ per run
- **Coverage**: Historical (several years)

### Using Local Data

To use your own crime dataset:
1. Save as CSV with columns: `date`, `latitude`, `longitude`, `crime_type`, `district`, `city`
2. Place in `data/` directory
3. Modify `src/data_loader.py` to load local file instead of API

```python
def load_local_data(filepath):
    df = pd.read_csv(filepath)
    # Preprocess...
    return df
```

## Next Steps

1. **Run the interactive dashboard** for exploration
2. **Configure for your city/jurisdiction**
3. **Train on your actual crime data** (if available)
4. **Validate predictions** against recent crime patterns
5. **Integrate with dispatch system** for operational use
6. **Monitor and retrain regularly** as crime patterns evolve

## Support Resources

- **Documentation**: See README.md
- **Examples**: Check `examples/` directory
- **Issues**: Review src/config.py for common settings
- **Community**: Check GitHub issues if available

## Quick Commands Reference

```bash
# Virtual environment
source venv/bin/activate          # Activate (Linux/macOS)
deactivate                        # Deactivate

# Running
python main.py --help             # Show options
streamlit run streamlit_app.py    # Start dashboard
python -m pytest                  # Run tests

# Data exploration
python src/data_loader.py         # Download and show data
python src/feature_engineering.py # Show feature samples

# Model management
python -c "from src.prediction_pipeline import CrimeHotspotPipeline; p = CrimeHotspotPipeline('Chicago'); p.save_model()"
```

---

**Ready to go!** Start with:
```bash
streamlit run streamlit_app.py
```
