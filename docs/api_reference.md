# API Reference: CrimeGuard-AI

## src/models.py
### `STGNN`
- **Description**: Spatial-Temporal Graph Neural Network for crime prediction.
- **Methods**:
    - `__init__(self, num_nodes, num_features, hidden_dim, seq_len, out_dim)`: Initializes spatial (GCN) and temporal (LSTM) layers.
    - `forward(self, x, edge_index)`: Executes spatial-temporal propagation.

## src/trainer.py
### `Trainer`
- **Description**: Manages training of the STGNN model.
- **Methods**:
    - `train(self, crime_df, weather_df, event_df, socio_df, coords, epochs)`: Orchestrates data pipeline, graph construction, and optimization loop.

## src/route_optimizer.py
### `optimize_patrol_route`
- **Description**: Solves the Traveling Salesperson Problem for optimal patrol paths.
- **Inputs**: `distance_matrix` (numpy array).
- **Outputs**: Ordered list of node indices for the patrol route.

## src/feature_engineering.py
### `FeatureEngineer`
- **Description**: Normalizes and merges external features.
- **Methods**:
    - `run_pipeline(self, crime_df, weather_df, event_df, socio_df)`: Integrated ingestion pipeline for all feature sets.

## src/graph_builder.py
### `build_edge_index`
- **Description**: Generates connectivity graph from geographic node coordinates.
- **Inputs**: `coords` (Nx2 array), `threshold` (connectivity limit).
- **Outputs**: `edge_index` tensor (2xN).

## src/inference.py
### `Predictor`
- **Description**: Loads serialized weights and computes forecasts.
- **Methods**:
    - `predict_batch(self, x, edge_index)`: Performs batch inference using loaded model weights.
