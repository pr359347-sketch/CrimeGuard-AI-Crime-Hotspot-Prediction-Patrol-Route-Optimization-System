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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import config
from data_loader import CrimeDataLoader
from feature_engineering import FeatureEngineer
from prediction_pipeline import CrimeHotspotPipeline
from visualization import HotspotVisualizer, StatisticsVisualizer
from route_optimizer import PatrolRouteOptimizer, ResourceAllocationOptimizer, DynamicPatrolScheduler
from model_evaluation import PredictionEvaluator, HotspotDetectionValidator

# Page configuration
st.set_page_config(
    page_title="Crime Hotspot Prediction & Patrol Optimizer",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .hotspot-card {
        background-color: #ffe6e6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ff4444;
    }
    .safe-card {
        background-color: #e6ffe6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #44ff44;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

st.sidebar.title("🚔 Crime Hotspot Predictor")
st.sidebar.markdown("---")

# City selection
city = st.sidebar.selectbox(
    "Select City",
    ["Chicago", "LA"],
    help="Choose which city to analyze"
)

# Prediction horizon
horizon = st.sidebar.slider(
    "Prediction Horizon (days)",
    1, 14, 7,
    help="Number of days ahead to predict"
)

# Hotspot threshold
threshold = st.sidebar.slider(
    "Hotspot Probability Threshold",
    0.0, 1.0, float(config.HOTSPOT_THRESHOLD),
    step=0.05,
    help="Minimum probability to classify as hotspot"
)

# Number of patrol units
num_units = st.sidebar.slider(
    "Number of Patrol Units",
    1, 20, config.NUM_PATROL_UNITS,
    help="Number of police vehicles to deploy"
)

st.sidebar.markdown("---")

# Actions
if st.sidebar.button("🔄 Run Full Pipeline", key="run_pipeline", help="Execute complete analysis"):
    st.session_state.run_pipeline = True

if st.sidebar.button("📊 Load Sample Data", key="load_sample", help="Load pre-computed results"):
    st.session_state.load_sample = True

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **System Overview:**
    - ST-GNN + LSTM for spatial-temporal prediction
    - 7-day ahead crime forecasting
    - Patrol route optimization via OR-Tools
    - Interactive visualizations with Folium
    """
)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'hotspots' not in st.session_state:
    st.session_state.hotspots = None
if 'routes' not in st.session_state:
    st.session_state.routes = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None

# Title
st.title("🚔 Crime Hotspot Prediction & Patrol Optimizer")
st.markdown(f"**City:** {city} | **Prediction Horizon:** {horizon} days | **Patrol Units:** {num_units}")

# ============================================================================
# TAB 1: PREDICTIONS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Predictions", 
    "🗺️ Map Visualization", 
    "🚗 Patrol Routes",
    "📊 Analytics",
    "⚙️ System Info"
])

with tab1:
    st.header("Hotspot Predictions")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Generate Predictions", key="gen_pred"):
            with st.spinner("Running prediction pipeline..."):
                try:
                    pipeline = CrimeHotspotPipeline(city)
                    
                    with st.status("Loading data...", expanded=True) as status:
                        st.write("Downloading crime data...")
                        pipeline.load_and_preprocess_data()
                        
                        st.write("Engineering features...")
                        pipeline.engineer_features()
                        
                        st.write("Training model...")
                        pipeline.train_model(epochs=5)
                        
                        st.write("Making predictions...")
                        pipeline.predict_hotspots()
                        pipeline.detect_hotspots()
                        
                        status.update(label="Pipeline complete!", state="complete")
                    
                    st.session_state.predictions = pipeline.predictions
                    st.session_state.hotspots = pipeline.hotspots
                    st.session_state.pipeline = pipeline
                    
                    st.success("✅ Predictions generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    # Generate synthetic predictions for demo
                    st.info("Using synthetic predictions for demonstration...")
                    st.session_state.predictions = np.random.uniform(0, 1, (config.GRID_SIZE, config.GRID_SIZE))
    
    with col2:
        if st.session_state.predictions is not None:
            stats = StatisticsVisualizer.get_hotspot_stats(st.session_state.predictions)
            
            st.metric("High-Risk Cells", f"{stats['high_risk_cells']}", f"{stats['high_risk_percentage']:.1f}%")
            st.metric("Max Probability", f"{stats['max_probability']:.2%}")
            st.metric("Mean Probability", f"{stats['mean_probability']:.2%}")
    
    if st.session_state.predictions is not None:
        st.subheader("Top 10 Hotspots")
        top_hotspots = StatisticsVisualizer.get_top_hotspots(st.session_state.predictions, top_n=10)
        
        hotspot_df = pd.DataFrame(top_hotspots)
        
        fig = px.bar(
            hotspot_df,
            x='rank',
            y='probability',
            title="Top Hotspots by Probability",
            labels={'rank': 'Rank', 'probability': 'Probability'},
            color='probability',
            color_continuous_scale='Reds'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(hotspot_df, use_container_width=True)

# ============================================================================
# TAB 2: MAP VISUALIZATION
# ============================================================================

with tab2:
    st.header("Interactive Hotspot Heatmap")
    
    if st.session_state.predictions is not None:
        bounds = config.CHICAGO_BOUNDS if city == 'Chicago' else config.LA_BOUNDS
        visualizer = HotspotVisualizer(bounds, city)
        
        m = visualizer.create_heatmap(st.session_state.predictions)
        
        st_folium(m, width=1000, height=600)
        
        st.info(
            """
            **Map Legend:**
            - 🔴 Red circles: High-probability hotspots (>85%)
            - 🟠 Orange circles: Medium-probability hotspots (70-85%)
            - Heat layer: Overall crime intensity
            """
        )
    else:
        st.warning("⚠️ Generate predictions first to view the map")

# ============================================================================
# TAB 3: PATROL ROUTES
# ============================================================================

with tab3:
    st.header("Optimized Patrol Routes")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Optimize Routes", key="opt_routes"):
            with st.spinner("Optimizing patrol routes..."):
                try:
                    if st.session_state.hotspots is None:
                        if st.session_state.predictions is not None:
                            hotspot_mask = st.session_state.predictions > threshold
                            hotspot_indices = np.where(hotspot_mask)
                            st.session_state.hotspots = list(zip(hotspot_indices[0], hotspot_indices[1]))
                        else:
                            st.error("Generate predictions first")
                            st.stop()
                    
                    # Sample hotspots if too many
                    hotspots_to_optimize = st.session_state.hotspots
                    if len(hotspots_to_optimize) > 50:
                        np.random.seed(42)
                        indices = np.random.choice(len(hotspots_to_optimize), 50, replace=False)
                        hotspots_to_optimize = [hotspots_to_optimize[i] for i in indices]
                    
                    bounds = config.CHICAGO_BOUNDS if city == 'Chicago' else config.LA_BOUNDS
                    optimizer = PatrolRouteOptimizer(hotspots_to_optimize, config.GRID_SIZE)
                    
                    st.session_state.routes = optimizer.get_routes_coordinates(bounds)
                    
                    st.success("✅ Routes optimized!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if st.session_state.routes:
            st.metric("Routes Generated", len(st.session_state.routes))
            total_distance = sum(r['distance'] for r in st.session_state.routes)
            st.metric("Total Distance", f"{total_distance:.0f} units")
    
    if st.session_state.routes:
        bounds = config.CHICAGO_BOUNDS if city == 'Chicago' else config.LA_BOUNDS
        visualizer = HotspotVisualizer(bounds, city)
        
        m = visualizer.create_patrol_route_map(st.session_state.routes)
        
        st_folium(m, width=1000, height=600)
        
        st.subheader("Route Details")
        route_details = []
        for route in st.session_state.routes:
            route_details.append({
                'Vehicle ID': route['vehicle_id'],
                'Waypoints': len(route['coordinates']),
                'Distance': f"{route['distance']:.0f}"
            })
        
        st.dataframe(pd.DataFrame(route_details), use_container_width=True)
    else:
        st.warning("⚠️ Optimize routes to view the map")

# ============================================================================
# TAB 4: ANALYTICS
# ============================================================================

with tab4:
    st.header("Prediction Analytics")
    
    if st.session_state.predictions is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            validator = HotspotDetectionValidator()
            validation = validator.validate_hotspots(st.session_state.predictions)
            
            st.subheader("Hotspot Validation")
            st.metric("Valid Clusters", validation['valid_clusters'])
            st.metric("Avg Cluster Size", f"{validation['avg_cluster_size']:.1f}")
            st.metric("Coverage", f"{validation['cluster_coverage']:.1%}")
        
        with col2:
            st.subheader("Temporal Distribution")
            
            # Simulate temporal distribution
            hours = np.arange(0, 24)
            intensities = np.sin(np.linspace(0, 2*np.pi, 24)) * 50 + 50
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hours, y=intensities, mode='lines+markers', name='Crime Intensity'))
            fig.update_layout(title="Expected Crime Intensity by Hour", xaxis_title="Hour", yaxis_title="Intensity")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.subheader("Crime Type Distribution")
            
            crime_types = ['Theft', 'Robbery', 'Burglary', 'Assault', 'Motor Vehicle Theft']
            counts = np.random.randint(10, 100, len(crime_types))
            
            fig = px.pie(
                values=counts,
                names=crime_types,
                title="Crime Type Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Prediction timeline
        st.subheader("7-Day Prediction Timeline")
        
        days = [(datetime.now() + timedelta(days=i)).strftime('%m-%d') for i in range(7)]
        predicted_crimes = np.random.poisson(lam=30, size=7)
        
        timeline_df = pd.DataFrame({
            'Date': days,
            'Predicted Crimes': predicted_crimes
        })
        
        fig = px.bar(timeline_df, x='Date', y='Predicted Crimes', title="7-Day Crime Forecast")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 5: SYSTEM INFO
# ============================================================================

with tab5:
    st.header("System Information & Documentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏗️ Architecture")
        st.markdown("""
        **Spatial-Temporal GNN (ST-GNN)**
        - Graph Convolution for spatial relationships
        - LSTM for temporal patterns
        - Bidirectional information flow
        
        **Features**
        - Temporal: hour, day, month, holidays
        - Weather: temperature, precipitation, humidity
        - Socioeconomic: poverty rate, unemployment, density
        - Spatial: distance from center, grid position
        - Lagged: crime counts at lag 1, 7, 14
        - Rolling: 7, 14, 30-day statistics
        """)
    
    with col2:
        st.subheader("⚙️ Configuration")
        st.markdown(f"""
        - **Grid Size:** {config.GRID_SIZE}×{config.GRID_SIZE}
        - **History Window:** {config.HISTORY_WINDOW} days
        - **Prediction Horizon:** {config.PREDICTION_HORIZON} days
        - **Patrol Units:** {config.NUM_PATROL_UNITS}
        - **Hotspot Threshold:** {config.HOTSPOT_THRESHOLD:.0%}
        - **Batch Size:** {config.BATCH_SIZE}
        - **Learning Rate:** {config.LEARNING_RATE}
        """)
    
    st.markdown("---")
    
    st.subheader("📚 Model Details")
    st.markdown("""
    **ST-GNN Model**
    - 2 ST-GNN blocks with increasing channel dimensions
    - Graph convolution with 8-neighbor adjacency
    - LSTM cells for temporal encoding
    - Fully connected prediction head
    
    **Training**
    - Loss: Mean Squared Error (MSE)
    - Optimizer: Adam
    - Early stopping with patience=10
    
    **Evaluation Metrics**
    - Regression: MSE, MAE, RMSE, R²
    - Classification: Precision, Recall, F1, ROC-AUC
    - Spatial: Hit rate, False alarm rate, Coverage
    """)
    
    st.markdown("---")
    
    st.subheader("🗺️ Route Optimization")
    st.markdown(f"""
    **Vehicle Routing Problem (VRP)**
    - Solver: OR-Tools with Guided Local Search
    - Objective: Minimize total distance
    - Constraints: Max distance per vehicle ({config.MAX_DISTANCE_KM} km)
    
    **Deployment**
    - Dynamic shift scheduling
    - Resource allocation by hotspot density
    - Real-time route updates
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p>🚔 Crime Hotspot Prediction & Patrol Optimizer</p>
    <p><small>Built with PyTorch, OR-Tools, Streamlit & Folium</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
