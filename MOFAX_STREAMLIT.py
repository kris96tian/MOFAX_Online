# mofa_streamlit_app.py
import streamlit as st
import mofax as mfx
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
st.set_page_config(page_title="MOFA+ Model Exploration", layout="wide")

# CSS Styling from HTML example
st.markdown("""
    <style>
    body { font-family: "Helvetica Neue", "Open Sans", sans-serif; -webkit-font-smoothing: antialiased; }
    h2 { font-size: 24pt; color: #0E2C37; padding: 20px 5px; }
    .container-fluid { background-color: #282C2F; color: white; }
    .tab-header { background-color: #F0F0F0; color: #555555; }
    .active-tab { background-color: white; color: #0E2C37; }
    .description { color: #999999; margin: 20px 0; }
    a { color: #999999; text-decoration: underline; }
    a:hover { color: white; }
    </style>
    """, unsafe_allow_html=True)

# Set up the Streamlit app title and sidebar
st.title("MOFA+ Model Exploration")
st.sidebar.header("Upload MOFA+ Model (.hdf5)")

# Upload model file
model_file = st.sidebar.file_uploader("Upload file", type=["hdf5"])

if model_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as tmp_file:
        tmp_file.write(model_file.read())
        temp_filepath = tmp_file.name

    # Initialize MOFA model with mofax
    m = mfx.mofa_model(temp_filepath)

    # Display basic model information
    st.subheader("Basic Model Features")
    st.write(f"**Cells**: {m.shape[0]}")
    st.write(f"**Features**: {m.shape[1]}")
    st.write(f"**Groups**: {', '.join(m.groups)}")
    st.write(f"**Views**: {', '.join(m.views)}")

    # Display HDF5 group structure and example data
    st.subheader("Model Weights")
    st.write("**HDF5 group structure:**")
    st.write(m.weights)

    st.write("**Sample Weight Data (as ndarray):**")
    st.write(m.get_weights()[:3, :5])

    st.write("**Sample Weight Data (as DataFrame):**")
    st.dataframe(m.get_weights(df=True).iloc[:3, :5])

    # Sidebar controls for interactivity
   # selected_view = st.sidebar.selectbox("Select View", m.views)
   # selected_group = st.sidebar.selectbox("Select Group", m.groups)
   # selected_factor = st.sidebar.selectbox("Select Factor", m.factors)

    # Sidebar input for number of features (for plot_weights_heatmap)
    n_features = st.sidebar.number_input("Select Number of Features to Display", min_value=1, max_value=20, value=5)

    # Interactive Data Analysis Features
    ## 1. Plot Top Feature Weights
    if st.checkbox("Plot Top Feature Weights"):
        st.subheader("Top Feature Weights")
        ax_weights = mfx.plot_weights_heatmap(
            m, 
            n_features=n_features,  # Use the dynamically set n_features
            factors=range(0, 10),
            xticklabels_size=6, 
            w_abs=True,
            cmap="viridis", 
            cluster_factors=False
        )
        st.pyplot(ax_weights.figure)

    if st.checkbox("Plot Weights correlation"):
        st.subheader("Weights correlation")
        try:
            ax_ranked = mfx.plot_weights_correlation(m)
            st.pyplot(ax_ranked.figure)
        except Exception as e:
            st.error(f"Failed to plot ranked weights: {e}")

    ## 4. Factor Loadings
    if st.checkbox("Plot Factor Loadings"):
        st.subheader("Factor Loadings")
        loadings = m.get_weights(df=True)
        fig_loadings = px.bar(loadings, title="Factor Loadings")
        st.plotly_chart(fig_loadings)

    ## 5. Correlation Matrix
    if st.checkbox("Show Factor Correlation Matrix"):
        st.subheader("Factor Correlation Matrix")
        correlation_matrix = m.get_weights(df=True).corr()
        fig_corr = px.imshow(correlation_matrix, title="Factor Correlation Matrix")
        st.plotly_chart(fig_corr)

    ## 6. Data Download Options
    st.sidebar.subheader("Download Data")
    if st.sidebar.button("Download Weights Data as CSV"):
        weights_df = m.get_weights(df=True)
        csv = weights_df.to_csv(index=False)
        st.sidebar.download_button(
            label="weights_data.csv",
            data=csv,
            file_name='weights_data.csv',
            mime='text/csv',
        )
    elif st.sidebar.button("Download Variance Explained as CSV"):
        dfff = m.calculate_variance_explained()
        csv = dfff.to_csv(index=False)
        st.sidebar.download_button(
            label="variance_explained.csv",
            data=csv,
            file_name='variance_explained.csv',
            mime='text/csv',
        )
else:
    st.write("Please upload a MOFA+ .hdf5 file to analyze.")
