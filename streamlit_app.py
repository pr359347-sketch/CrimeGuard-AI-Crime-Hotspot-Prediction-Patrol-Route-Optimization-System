import streamlit as st
import torch
import numpy as np
import folium
from streamlit_folium import st_folium
import os

from src.inference import Predictor
from src.route_optimizer import optimize_patrol_route
from src.graph_builder import build_edge_index

# Configuration
from pathlib import Path

MODEL_PATH = Path("models") / "test_weights.pth"
NUM_NODES = 100
NUM_FEATURES = 5
HIDDEN_DIM = 16
SEQ_LEN = 7

# Generate mock coordinates for 100 nodes
np.random.seed(42)
node_coords_arr = np.random.rand(NUM_NODES, 2)
node_coords = {i: [41.8781 + node_coords_arr[i, 0]*0.1, -87.6298 + node_coords_arr[i, 1]*0.1] for i in range(NUM_NODES)}

@st.cache_resource
def load_predictor():
    if not os.path.exists(MODEL_PATH):
        from src.models import STGNN
        model = STGNN(NUM_NODES, NUM_FEATURES, HIDDEN_DIM, SEQ_LEN, out_dim=1)
        torch.save(model.state_dict(), MODEL_PATH)
    return Predictor(MODEL_PATH, NUM_NODES, NUM_FEATURES, HIDDEN_DIM, SEQ_LEN)

@st.cache_resource
def get_edge_index():
    return build_edge_index(node_coords_arr, threshold=0.05)

st.set_page_config(page_title="CrimeGuard-AI", layout="wide")

st.title("CrimeGuard-AI Dashboard")
st.markdown("Predictive crime hotspot analysis and patrol optimization.")

# Run Inference
predictor = load_predictor()
edge_index = get_edge_index()

# Mock socioeconomic features (e.g., median income by node)
socio_features = torch.randn(1, NUM_NODES, 1) # Added as a dummy feature dimension
dummy_input = torch.cat([torch.randn(1, SEQ_LEN, NUM_NODES, NUM_FEATURES-1), socio_features.unsqueeze(1).repeat(1, SEQ_LEN, 1, 1)], dim=-1)

predictions = predictor.predict_batch(dummy_input, edge_index)
predictions = predictions.detach().cpu().numpy()

# Map Integration
m = folium.Map(location=[41.8781, -87.6298], zoom_start=13)

# Filter high intensity nodes and compute route
hotspot_indices = []

for i in range(predictions.shape[1]):
    score = float(predictions[0][i][0])
    if score > 0.5:
        hotspot_indices.append(i)
if hotspot_indices:
    coords = [node_coords[i] for i in hotspot_indices]
    dist_matrix = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)):
        for j in range(len(coords)):
            dist_matrix[i, j] = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
    
    route_indices = optimize_patrol_route(dist_matrix)
    
    if route_indices:
        route_coords = [coords[i] for i in route_indices]
        folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.8).add_to(m)

# Render Hotspots
for i in hotspot_indices:
    folium.CircleMarker(
        location=node_coords[i],
        radius=float(predictions[0, i]) * 20,
        color='red',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

st_folium(
    m,
    width=1000,
    height=500,
    returned_objects=[]
)
