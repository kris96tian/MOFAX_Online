# MOFA+ Model Explorer 🌐📊

Explore, visualize, and analyze your **MOFA+ (Multi-Omics Factor Analysis)** results with ease! This Streamlit app provides an interactive interface to load and interrogate **multi-omics models** in `.hdf5` format.

## 🚀 Features

### Data Analysis
- **Upload Models**: Analyze MOFA+ models via `.hdf5` uploads
- **Intuitive Metrics**: Displays total cells, features, and groups at a glance
- **Enrichment Analysis**: 
  - Pathway and GO term enrichment for top features
  - Interactive filtering by factors and sources
  - Downloadable enrichment results

### Visualizations
- **Feature Analysis**:
  - Top feature weights heatmap
  - Ranked weight plots
- **Statistical Insights**:
  - Factor correlations (Pearson)
  - Variance explained analysis
- **Enrichment Plots**:
  - Factor-specific enrichment visualizations
  - Significance bar plots

### Data Export
- Download weights data as CSV
- Export variance explained metrics
- Save enrichment analysis results

### Customization
- Adjust analysis parameters
- Control feature selection counts
- Filter enrichment results by source
- Customize visualization options

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/kris96tian/mofa-explorer.git

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🖥️ Try it Live

Access the app at **[MOFA+ Model Explorer](https://mofaviz.streamlit.app/)**.

## 🧬 About MOFA+

The app leverages MOFA+ for **integrative analysis** of multi-modal single-cell data. The enrichment analysis feature helps identify biological pathways and processes associated with MOFA factors. To learn more, see:

Argelaguet, R., Arnol, D., Bredikhin, D., et al. **MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data.** *Genome Biology*, 21, 111 (2020). [DOI:10.1186/s13059-020-02015-1](https://doi.org/10.1186/s13059-020-02015-1)

## 📊 Example Usage

1. Upload your MOFA+ model (.hdf5 file)
2. View basic model metrics and statistics
3. Explore feature weights and correlations
4. Run enrichment analysis on factors of interest
5. Export results for further analysis

---

## ✨ Creator

Developed by **Kristian Alikaj**. Connect on:
- **[GitHub](https://github.com/kris96tian)**
- **[Portfolio](https://kris96tian.github.io/)**
