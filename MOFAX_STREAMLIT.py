import streamlit as st
import mofax as mfx
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import warnings
import numpy as np
import seaborn as sns
from gprofiler import GProfiler

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MOFA+ Model Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main layout and typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #1a1f36;
    }
    
    /* Custom title styling */
    .custom-title {
        background: linear-gradient(120deg, #2b5876, #4e4376);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card-like containers */
    .stDataFrame {
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 1rem;
        background: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e1e4e8;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #1a1f36;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #2b5876, #4e4376);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f1f5f9;
        padding: 2rem 1rem;
    }
    
    /* Plot containers */
    .plot-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="custom-title">
        <h1 style='margin:0'>MOFA+ Model Explorer</h1>
        <p style='margin:0;opacity:0.8;font-size:1.1em'>Analyze and visualize multi-omics factor analysis results</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Model Configuration")
    model_file = st.file_uploader("Upload MOFA+ Model (.hdf5)", type=["hdf5"])
    if st.button("Run Glioblastoma Model"):
        model_file = "model.hdf5"  
    if st.button("Run Breast Cancer Model"):
        model_file = "model_br.hdf5"  

if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as tmp_file:
        if isinstance(model_file, str): 
            temp_filepath = model_file
        else:
            tmp_file.write(model_file.read())
            temp_filepath = tmp_file.name

    m = mfx.mofa_model(temp_filepath)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"<div class='metric-container'><div class='metric-label'>Total Cells</div><div class='metric-value'>{m.shape[0]:,}</div></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<div class='metric-container'><div class='metric-label'>Features</div><div class='metric-value'>{m.shape[1]:,}</div></div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"<div class='metric-container'><div class='metric-label'>Groups</div><div class='metric-value'>{len(m.groups)}</div></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Analysis Parameters")
        weights_df = m.get_weights()
        weights_df = pd.DataFrame(weights_df)
        selected_factor = st.selectbox("Select Factor", weights_df.columns)
        n_features = st.slider("Number of Features to Display", 
                             min_value=1, 
                             max_value=20, 
                             value=5,
                             help="Adjust the number of features shown in visualizations")

        st.markdown("### Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Weights Data"):
                st.dataframe(weights_df)
                csv = weights_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Weights CSV",
                    data=csv,
                    file_name='weights.csv',
                    mime='text/csv'
                )

        with col2:
            if st.button("📥 Enrichment Results"):
                weights_df = pd.DataFrame(weights_df)
                top_features = get_top_features(weights_df, n_features)
                enrichment_results = run_enrichment(top_features)
                st.dataframe(enrichment_results)
                csv = enrichment_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Enrichment CSV",
                    data=csv,
                    file_name='enrichment_results.csv',
                    mime='text/csv'
                )

    st.markdown("### Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Feature Weights")
        if selected_factor:
            factor_weights = weights_df[selected_factor]
            fig = px.bar(
                x=factor_weights.index,
                y=factor_weights.values,
                labels={'x': 'Feature', 'y': 'Weight'},
                title=f"Feature Weights for {selected_factor}"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Enrichment Analysis")
        if selected_factor:
            top_features = get_top_features(weights_df, n_features)
            enrichment_results = run_enrichment(top_features)
            fig = plot_enrichment(selected_factor, enrichment_results)
            if fig:
                st.pyplot(fig)

    st.markdown("### Explore Factor Data")
    if st.checkbox("Show Raw Factor Data"):
        factor_data = m.get_factors()
        st.dataframe(factor_data)

        csv = factor_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Factor Data CSV",
            data=csv,
            file_name='factor_data.csv',
            mime='text/csv'
        )
else:
    st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h2 style="color: #1a1f36; margin-bottom: 1rem;">Welcome to MOFA+ Model Explorer</h2>
            <p style="color: #6b7280; font-size: 1.1em;">Please upload a MOFA+ .hdf5 file using the sidebar to begin your analysis.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
        ---
        **Created by Kristian Alikaj**  
        For more, visit [My GitHub](https://github.com/kris96tian) or [My Portfolio Website](https://kris96tian.github.io/)
""")

