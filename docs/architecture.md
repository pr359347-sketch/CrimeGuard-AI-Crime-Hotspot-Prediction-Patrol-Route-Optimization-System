# CrimeGuard-AI Architecture Documentation

## Overview
CrimeGuard-AI is a predictive intelligence platform for urban safety. It utilizes a multi-layered pipeline to transform raw environmental and socio-economic data into actionable patrol intelligence.

## 1. Feature Engineering Pipeline
The feature pipeline (`src/feature_engineering.py`) acts as the primary data ingestion and normalization layer. It performs:
- **Temporal Alignment:** Merges crime incidents, weather reports, and event schedules.
- **Socioeconomic Enrichment:** Integrates static regional features (e.g., income/population density) via `region_id` mapping.
- **Data Cleaning:** Imputes missing values to maintain high-quality input tensors for the ML model.

## 2. Spatial-Temporal Forecasting (ST-GNN)
The core model (`src/models.py`) is based on a Spatial-Temporal Graph Neural Network architecture:
- **Spatial Layer:** Uses `GCNConv` (Graph Convolutional Networks) via `torch_geometric` to model geographical correlations between neighborhoods based on a dynamically built adjacency matrix (`src/graph_builder.py`).
- **Temporal Layer:** Employs an `LSTM` block to process sequence-based crime patterns, effectively capturing temporal dependencies.
- **Output:** Produces a 7-day crime probability forecast.

## 3. Inference & Optimization Flow
The pipeline transitions from forecasting to optimization:
- **Inference (`src/inference.py`):** A streamlined `Predictor` class loads serialized model weights (`models/`) to compute batch crime hotspots.
- **Route Optimization (`src/route_optimizer.py`):** Identified high-probability nodes are passed into an OR-Tools TSP solver, which generates the most efficient route covering these locations.
- **Visualization:** The optimized patrol route and predicted hotspots are rendered as an interactive overlay on the `streamlit_app.py` dashboard.
