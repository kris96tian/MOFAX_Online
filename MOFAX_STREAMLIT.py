import streamlit as st
import mofax as mfx
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import warnings
warnings.filterwarnings("ignore")

# Page configuration with custom theme
st.set_page_config(
    page_title="MOFA+ Model Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern styling
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

# Custom title with gradient background
st.markdown("""
    <div class="custom-title">
        <h1 style='margin:0'>MOFA+ Model Explorer</h1>
        <p style='margin:0;opacity:0.8;font-size:1.1em'>Analyze and visualize multi-omics factor analysis results</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with improved styling
with st.sidebar:
    st.markdown("### Model Configuration")
    model_file = st.file_uploader("Upload MOFA+ Model (.hdf5)", type=["hdf5"])

if model_file:
    # Save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as tmp_file:
        tmp_file.write(model_file.read())
        temp_filepath = tmp_file.name

    # Initialize MOFA model
    m = mfx.mofa_model(temp_filepath)

    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-container">
                <div class="metric-label">Total Cells</div>
                <div class="metric-value">{:,}</div>
            </div>
        """.format(m.shape[0]), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-container">
                <div class="metric-label">Features</div>
                <div class="metric-value">{:,}</div>
            </div>
        """.format(m.shape[1]), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-container">
                <div class="metric-label">Groups</div>
                <div class="metric-value">{}</div>
            </div>
        """.format(len(m.groups)), unsafe_allow_html=True)

    # Sidebar controls with improved organization
    with st.sidebar:
        st.markdown("### Analysis Parameters")
        selected_factor = st.selectbox("Select Factor", m.factors)
        n_features = st.slider("Number of Features to Display", 
                             min_value=1, 
                             max_value=20, 
                             value=5,
                             help="Adjust the number of features shown in visualizations")

        st.markdown("### Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Weights Data"):
                weights_df = m.get_weights(df=True)
                csv = weights_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='weights_data.csv',
                    mime='text/csv',
                )
        
        with col2:
            if st.button("📊 Variance Data"):
                variance_df = m.calculate_variance_explained()
                csv = variance_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=variance_df.to_csv(index=False),
                    file_name='variance_explained.csv',
                    mime='text/csv',
                )

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Weights", "Ranked Weights", "Variance Analysis", "Correlation Matrix"])

    with tab1:
        st.markdown("### Top Feature Weights")
        with st.container():
            ax_weights = mfx.plot_weights_heatmap(
                m, 
                n_features=n_features,  
                factors=range(0, 10),
                xticklabels_size=6, 
                w_abs=True,
                cmap="viridis"
            )
            plt.tight_layout()
            st.pyplot(ax_weights.figure)

    with tab2:
        st.markdown("### Weights Ranked by Factor")
        try:
            ax_ranked = mfx.plot_weights_ranked(
                m, 
                factor=selected_factor, 
                n_features=n_features,
                y_repel_coef=0.04, 
                x_rank_offset=-150
            )
            plt.tight_layout()
            st.pyplot(ax_ranked.figure)
        except Exception as e:
            st.error(f"Failed to plot ranked weights: {str(e)}")

    with tab3:
        st.markdown("### Variance Explained Analysis")
        variance_df = m.get_r2(factors=list(range(selected_factor))).sort_values("R2", ascending=False)
        
        # Create a bar chart for variance explained
        fig = px.bar(variance_df.head(10), 
                    y='R2', 
                    title='Top 10 Variance Explained by Factor',
                    labels={'R2': 'R² Value'},
                    template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Detailed Data"):
            st.dataframe(variance_df)

    with tab4:
        st.markdown("### Factor Correlation Analysis")
        correlation_matrix = m.get_weights(df=True).corr()
        fig_corr = px.imshow(
            correlation_matrix,
            title="Factor Correlation Matrix",
            color_continuous_scale="RdBu",
            template="plotly_white"
        )
        fig_corr.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=600
        )
        st.plotly_chart(fig_corr, use_container_width=True)

else:
    # Welcome message with improved styling
    st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h2 style="color: #1a1f36; margin-bottom: 1rem;">Welcome to MOFA+ Model Explorer</h2>
            <p style="color: #6b7280; font-size: 1.1em;">Please upload a MOFA+ .hdf5 file using the sidebar to begin your analysis.</p>
        </div>
    """, unsafe_allow_html=True)
# Footer
st.markdown("""
        ---
        **Created by Kristian Alikaj**  
        For more, visit [My GitHub](https://github.com/kris96tian) or [My Portfolio Website](https://kris96tian.github.io/)
    """)

