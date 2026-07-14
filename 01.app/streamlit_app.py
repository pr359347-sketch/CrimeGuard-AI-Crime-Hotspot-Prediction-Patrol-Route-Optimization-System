"""Streamlit Dashboard for Crime Hotspot Prediction & Patrol Optimizer"""
import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '05.src', 'src'))

from src.config import config
from src.data_loader import CrimeDataLoader
from src.feature_engineering import FeatureEngineer
from src.prediction_pipeline import CrimeHotspotPipeline
from src.visualization import HotspotVisualizer, StatisticsVisualizer
from src.route_optimizer import PatrolRouteOptimizer, ResourceAllocationOptimizer, DynamicPatrolScheduler
from src.model_evaluation import PredictionEvaluator, HotspotDetectionValidator

# Page configuration
st.set_page_config(
    page_title="Crime Hotspot Prediction & Patrol Optimizer",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ... (truncated)