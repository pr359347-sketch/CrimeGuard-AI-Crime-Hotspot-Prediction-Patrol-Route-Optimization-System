<<<<<<< HEAD
# 🚔 Crime Hotspot Prediction & Patrol Optimizer

A comprehensive spatial-temporal machine learning system for predicting crime hotspots 7 days in advance and optimizing police patrol routes using graph neural networks and vehicle routing optimization.

## 🎯 Overview

This project combines:
- **Spatial-Temporal GNN (ST-GNN)** with LSTM for predicting crime hotspots
- **Graph Neural Networks** for capturing spatial relationships
- **Vehicle Route Optimization** using OR-Tools for efficient patrol planning
- **Interactive Visualizations** with Folium and Streamlit
- **Real-world Data** from Chicago and LA crime datasets

## 🏗️ Architecture

### Machine Learning Stack
```
Data Layer
    ↓
Feature Engineering (Temporal, Weather, Socioeconomic, Spatial)
    ↓
Spatial-Temporal GNN Model
    ├─ Graph Convolution (Spatial)
    └─ LSTM (Temporal)
    ↓
Hotspot Prediction (7-day forecast)
    ↓
Route Optimization (OR-Tools VRP)
    ↓
Visualizations & Dashboard
```

### Core Components

#### 1. **Data Loading** (`data_loader.py`)
- Downloads from Chicago/LA open data APIs (Socrata)
- Handles missing values and geographic filtering
- Creates spatial grid representation (50×50 cells)
- Temporal aggregation by day

#### 2. **Feature Engineering** (`feature_engineering.py`)
- **Temporal Features**: hour, day, month, holidays, cyclic encoding
- **Weather Features**: temperature, precipitation, humidity, wind speed
- **Socioeconomic**: poverty rate, unemployment, population density, median income
- **Spatial Features**: distance from center, grid position
- **Lagged Features**: crime counts at lag 1, 7, 14 days
- **Rolling Statistics**: 7, 14, 30-day means and standard deviations

#### 3. **Spatial-Temporal GNN Model** (`st_gnn_model.py`)
```python
STGNNBlock
├── SpatialGraphConv (GCN layer)
├── TemporalLSTM (LSTM layer)
└── Batch Normalization & Dropout

Complete Model
├── ST-GNN Block 1 (32 channels)
├── ST-GNN Block 2 (64 channels)
└── Prediction Head (FC layers)
```

#### 4. **Route Optimization** (`route_optimizer.py`)
- Vehicle Routing Problem (VRP) solver
- OR-Tools with Guided Local Search metaheuristic
- Dynamic patrol scheduling by shift
- Resource allocation based on hotspot density

#### 5. **Visualization** (`visualization.py`)
- Interactive Folium heatmaps
- Patrol route mapping
- Hotspot comparison maps
- Statistical summaries

#### 6. **Evaluation** (`model_evaluation.py`)
- Regression metrics: MSE, MAE, RMSE, R²
- Classification metrics: Precision, Recall, F1, ROC-AUC
- Spatial metrics: Hit rate, False alarm rate, Coverage
- Hotspot validation and clustering analysis

#### 7. **Complete Pipeline** (`prediction_pipeline.py`)
- End-to-end orchestration
- Model training and saving
- Comprehensive evaluation
- Visualization generation

## 📊 Data

### Sources
- **Chicago**: https://data.cityofchicago.org/api/views/ijzp-q8t2 (Crimes dataset)
- **LA**: https://data.lacity.gov/api/views/2nrs-mtv8 (Crime data)

### Features (30+)
- **Temporal** (8): hour, day, weekday, month, weekend flag, holiday flag, cyclic encodings
- **Weather** (4): temperature, precipitation, humidity, wind speed
- **Socioeconomic** (5): poverty rate, unemployment, density, income, education
- **Spatial** (5): lat/lon normalized, distance from center, grid position
- **Lagged** (3): crime counts at t-1, t-7, t-14
- **Rolling** (6): 7/14/30-day means and standard deviations

## 🚀 Quick Start

### 1. Installation

```bash
cd crime_hotspot_optimizer
pip install -r requirements.txt
```

### 2. Run Command-Line Interface

```bash
# Full pipeline (predict + optimize + visualize)
python main.py --city Chicago --mode full --epochs 10

# Only prediction
python main.py --city Chicago --mode predict --epochs 50

# Only route optimization
python main.py --city Chicago --mode optimize

# Only visualizations
python main.py --city Chicago --mode visualize
```

### 3. Launch Interactive Dashboard

```bash
streamlit run streamlit_app.py
```

Then open: http://localhost:8501

## 📈 Model Performance

### Expected Metrics
- **Regression R²**: 0.65-0.75
- **Classification F1**: 0.60-0.70
- **Spatial Hit Rate**: 0.55-0.65
- **False Alarm Rate**: <0.30

### Optimization Results
- **Route Distance Reduction**: 25-35% vs baseline
- **Coverage**: 70-80% of crimes in high-risk areas
- **Deployment Efficiency**: Optimal vehicle utilization

## 🗺️ Configuration

Edit `src/config.py` to customize:

```python
GRID_SIZE = 50                    # Spatial grid resolution
PREDICTION_HORIZON = 7            # Days ahead
HISTORY_WINDOW = 60               # Days history
ST_GNN_HIDDEN_DIM = 64            # GNN hidden dimension
LSTM_HIDDEN_DIM = 128             # LSTM hidden dimension
NUM_PATROL_UNITS = 10             # Number of vehicles
HOTSPOT_THRESHOLD = 0.7           # Detection threshold
```

## 📁 Project Structure

```
crime_hotspot_optimizer/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration & constants
│   ├── data_loader.py            # Data loading & preprocessing
│   ├── feature_engineering.py    # Feature extraction
│   ├── st_gnn_model.py          # ST-GNN model architecture
│   ├── route_optimizer.py       # Route optimization engine
│   ├── visualization.py          # Visualization utilities
│   ├── model_evaluation.py       # Evaluation metrics
│   └── prediction_pipeline.py    # Complete pipeline
├── data/                         # Raw & processed data
├── models/                       # Trained models
├── outputs/                      # Results & visualizations
├── streamlit_app.py              # Interactive dashboard
├── main.py                       # CLI entry point
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🎮 Dashboard Features

### 🎯 Predictions Tab
- Generate crime hotspot predictions
- View top 10 hotspots by probability
- Real-time statistics

### 🗺️ Map Visualization Tab
- Interactive Folium heatmap
- Color-coded hotspot intensity
- Circular markers for high-probability areas

### 🚗 Patrol Routes Tab
- Optimized vehicle routes
- Route distance and efficiency metrics
- Waypoint-by-waypoint navigation

### 📊 Analytics Tab
- Hotspot validation metrics
- Temporal intensity distribution
- Crime type breakdown
- 7-day crime forecast

### ⚙️ System Info Tab
- Model architecture details
- Configuration parameters
- Route optimization explanation

## 🔬 Model Details

### ST-GNN Architecture

**Graph Convolution Layer**
```
Spatial Relationship Modeling
A = Adjacency Matrix (8-neighbor grid)
X = Node Features
Z = (A @ X) @ W + b
```

**LSTM Layer**
```
Temporal Pattern Learning
h_t = LSTM(X_t, h_{t-1})
```

**Combined ST-GNN Block**
```
1. Apply spatial graph convolution at each time step
2. Apply ReLU + Dropout
3. Apply LSTM on temporal dimension
4. Output: Spatial-temporal features
```

### Training Strategy
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: None (early stopping instead)
- **Early Stopping**: Patience=10 epochs
- **Batch Size**: 32
- **Epochs**: 50 (configurable)

### Route Optimization
- **Solver**: OR-Tools
- **Metaheuristic**: Guided Local Search
- **Objective**: Minimize total distance
- **Constraints**: Max distance per vehicle (50 km)
- **Time Limit**: 5 seconds

## 📊 Data Processing Pipeline

```
Raw Crime Data (10,000+ records)
    ↓
Spatial-Temporal Aggregation (50×50 grid, daily)
    ↓
Feature Engineering (30+ features)
    ↓
Sequence Creation (sliding windows)
    ↓
Train/Val/Test Split (80/10/10)
    ↓
Model Training & Evaluation
    ↓
Hotspot Detection & Post-processing
    ↓
Route Optimization & Visualization
```

## 🎯 Use Cases

1. **Police Department Planning**
   - Allocate patrol units efficiently
   - Reduce response times
   - Prevent crime through deterrence

2. **Urban Planning**
   - Identify high-risk neighborhoods
   - Plan community interventions
   - Monitor temporal patterns

3. **Resource Management**
   - Optimize personnel scheduling
   - Reduce operational costs
   - Improve coverage

4. **Research & Analysis**
   - Study spatial-temporal crime patterns
   - Evaluate intervention effectiveness
   - Generate insights for policy

## ⚠️ Limitations & Considerations

- **Data Quality**: Depends on accurate crime reporting
- **Bias**: May reflect policing patterns rather than actual crime
- **Seasonality**: Requires sufficient historical data (60+ days)
- **Generalization**: Model trained on specific city/period
- **Temporal Decay**: Predictions less accurate beyond 7 days

## 🔮 Future Improvements

- [ ] Multi-modal data integration (social media, events, weather APIs)
- [ ] Uncertainty quantification (Bayesian approach)
- [ ] Hierarchical clustering for dynamic region definition
- [ ] Real-time model updates with streaming data
- [ ] Fairness constraints in optimization
- [ ] Comparative analysis of different GNN architectures
- [ ] Mobile app for patrol officers

## 📚 References

### Papers
- Kipf & Welling (2016) - Semi-Supervised Classification with GCNs
- Hochreiter & Schmidhuber (1997) - LSTM
- Yao et al. (2018) - Deep Spatio-Temporal Residual Networks
- Jia et al. (2021) - Graph Attention Networks for Spatio-Temporal Data

### Libraries
- PyTorch: Deep learning framework
- PyTorch Geometric: GNN implementations
- OR-Tools: Vehicle routing solver
- Folium: Interactive maps
- Streamlit: Dashboard framework

## 📝 Usage Examples

### Example 1: Quick Prediction
```python
from src.prediction_pipeline import CrimeHotspotPipeline

pipeline = CrimeHotspotPipeline('Chicago')
pipeline.load_and_preprocess_data()
pipeline.engineer_features()
pipeline.train_model(epochs=20)
predictions = pipeline.predict_hotspots()
hotspots = pipeline.detect_hotspots()
```

### Example 2: Route Optimization
```python
from src.route_optimizer import PatrolRouteOptimizer

optimizer = PatrolRouteOptimizer(hotspots)
routes = optimizer.get_routes_coordinates(bounds)
for route in routes:
    print(f"Vehicle {route['vehicle_id']}: {route['distance']} units")
```

### Example 3: Visualization
```python
from src.visualization import HotspotVisualizer

visualizer = HotspotVisualizer(bounds, 'Chicago')
heatmap = visualizer.create_heatmap(predictions, 'output.html')
route_map = visualizer.create_patrol_route_map(routes, 'routes.html')
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Better feature engineering
- Alternative GNN architectures
- Hyperparameter optimization
- Additional data sources
- Performance optimizations

## 📄 License

MIT License - See LICENSE file for details

## 📞 Support

For issues or questions:
1. Check the documentation
2. Review example usage
3. Open an issue with detailed description

---

**Built with ❤️ for civic safety and data-driven policing**

Last Updated: 2026
Version: 1.0.0
=======
# CrimeGuard-AI-Crime-Hotspot-Prediction-Patrol-Route-Optimization-System
Tech Stack  Python | PyTorch | PyTorch Geometric | ST-GNN | LSTM | OR-Tools | Folium | Streamlit | Pandas | NumPy | Scikit-Learn AI-powered crime hotspot forecasting and patrol route optimization using ST-GNN, LSTM, OR-Tools, Folium, and Streamlit.
# 🚔 CrimeGuard AI: Crime Hotspot Prediction & Patrol Route Optimization

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![ST-GNN](https://img.shields.io/badge/ST--GNN-Spatial%20AI-green)
![LSTM](https://img.shields.io/badge/LSTM-Time%20Series-orange)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

> An AI-powered public safety intelligence platform designed to forecast crime hotspots 7 days in advance and optimize police patrol routes using Spatial-Temporal Deep Learning and Operations Research techniques.

---

# 📌 Project Overview

Crime remains one of the most critical challenges for modern cities. Traditional policing approaches are often reactive, responding to incidents after they occur rather than preventing them.

**CrimeGuard AI** aims to transform public safety through predictive analytics by leveraging historical crime patterns, weather conditions, public events, and socioeconomic indicators to forecast crime hotspots before incidents occur.

The platform will combine Spatial-Temporal Graph Neural Networks (ST-GNN), LSTM-based forecasting, and route optimization algorithms to help law enforcement agencies make proactive and data-driven decisions.

---

# 🎯 Project Objectives

The primary objectives of this project are:

- Predict crime hotspots up to 7 days in advance.
- Analyze spatial and temporal crime patterns.
- Incorporate weather, event, and socioeconomic factors into crime forecasting.
- Identify high-risk regions using AI-driven risk scoring.
- Generate interactive crime heatmaps for decision support.
- Optimize patrol deployment using route optimization algorithms.
- Support proactive policing and resource allocation.

---

# 🌎 Real-World Impact

This project addresses a real-world civic challenge by helping law enforcement agencies:

✅ Anticipate crime-prone areas before incidents occur

✅ Improve patrol efficiency

✅ Reduce response times

✅ Optimize resource utilization

✅ Enable data-driven public safety strategies

---

# 📊 Data Sources

The project will integrate multiple datasets to improve forecasting accuracy.

### Crime Data

- Chicago Crime Dataset
- Los Angeles Crime Dataset

### Weather Data

- Temperature
- Rainfall
- Humidity
- Wind Speed

### Event Data

- Festivals
- Concerts
- Sports Events
- Public Gatherings

### Socioeconomic Data

- Population Density
- Median Income
- Employment Indicators
- Community Demographics

---

# 🧠 AI & Machine Learning Architecture

The project combines spatial learning and temporal forecasting.

### Spatial Component

**Spatial-Temporal Graph Neural Network (ST-GNN)**

Responsible for:

- Learning neighborhood crime relationships
- Understanding spatial dependencies
- Modeling interactions between adjacent regions

### Temporal Component

**Long Short-Term Memory (LSTM)**

Responsible for:

- Learning historical crime trends
- Capturing temporal crime patterns
- Forecasting future crime activity

### Prediction Output

The model will forecast:

- Crime Risk Score
- Hotspot Probability
- Predicted Crime Volume
- Risk Classification

---

# 🏙️ Spatial Grid System

The city will be divided into multiple geographic grids.

Example:

```text
A1  A2  A3
B1  B2  B3
C1  C2  C3
```

Each grid will represent a graph node.

Adjacent grids will form graph connections used by the ST-GNN model.

---

# 📈 Feature Engineering

Features planned for model training:

### Historical Crime Features

- Crime Count (1 Day)
- Crime Count (3 Days)
- Crime Count (7 Days)
- Crime Type Distribution

### Weather Features

- Temperature
- Rainfall
- Humidity
- Wind Speed

### Event Features

- Event Count
- Festival Count
- Sports Events
- Public Gatherings

### Socioeconomic Features

- Population Density
- Median Income
- Community Risk Index

---

# 🚓 Patrol Route Optimization

To improve patrol effectiveness, the project will integrate:

### Google OR-Tools

The optimization engine will:

- Identify highest-risk locations
- Generate optimal patrol routes
- Minimize travel distance
- Maximize hotspot coverage

### Output

```text
Police Station
      ↓
Hotspot A
      ↓
Hotspot B
      ↓
Hotspot C
```

---

# 🗺️ Visualization & Dashboard

### Interactive Crime Heatmaps

Built using:

- Folium
- Leaflet Maps

Features:

- Crime Density Visualization
- Hotspot Highlighting
- Risk Layer Mapping
- Geographic Exploration

### Streamlit Dashboard

Dashboard modules:

- Crime Overview
- Hotspot Forecasting
- Risk Analytics
- Patrol Route Visualization
- Trend Analysis
- Model Performance Monitoring

---

# 🔬 Explainable AI

To improve transparency and trust, the project will include:

### SHAP Explainability

Used to understand:

- Why a region is predicted as high-risk
- Most influential factors
- Feature-level impact analysis

---

# 🛠️ Tech Stack

### Programming

- Python

### Machine Learning & Deep Learning

- PyTorch
- PyTorch Geometric
- ST-GNN
- LSTM

### Data Analysis

- Pandas
- NumPy
- Scikit-Learn

### Optimization

- Google OR-Tools

### Visualization

- Folium
- Matplotlib
- Seaborn

### Dashboard

- Streamlit

### Version Control

- Git
- GitHub

---

# 📂 Planned Project Structure

```text
CrimeGuard-AI-Crime-Hotspot-Prediction

├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Feature_Engineering.ipynb
│
├── models/
│   └── stgnn_lstm_model.pt
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── graph_builder.py
│   ├── train_model.py
│   ├── predict_hotspots.py
│   ├── route_optimizer.py
│   └── heatmap_generator.py
│
├── images/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# 🚀 Future Enhancements

- Real-Time Crime Monitoring
- Multi-City Crime Forecasting
- Crime Category Prediction
- SHAP-Based Explainability
- Real-Time Alert System
- Cloud Deployment
- Mobile-Friendly Dashboard
- Law Enforcement Decision Support System

---

# 🌟 Expected Outcomes

✔ Crime Hotspot Forecasting (7-Day Horizon)

✔ Spatial-Temporal Deep Learning Pipeline

✔ Interactive Crime Heatmaps

✔ Patrol Route Optimization

✔ Explainable AI Insights

✔ Streamlit Decision Support Dashboard

✔ Real-World Public Safety Use Case

---

# 👩‍💻 Author

**Priya Rani**

B.Tech Computer Science Engineering (2023–2027)

Passionate about Artificial Intelligence, Machine Learning, Data Analytics, and building data-driven solutions that create real-world impact.

---

## ⭐ This project is currently under development. Contributions, feedback, and suggestions are welcome.
>>>>>>> 1eaa5a02ad92c1fafd2955623b269fb9517ca0ee
