# 🚗 Cars Dataset - Comprehensive Exploratory Data Analysis (EDA)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Key Research Question](#key-research-question)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Results](#analysis-results)
- [Improvements Made](#improvements-made)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project provides a comprehensive exploratory data analysis (EDA) of a cars dataset, investigating the relationship between car choices (brand, model, color) and customers' financial profiles (credit card types). The analysis includes statistical testing, advanced visualizations, geographic mapping, and machine learning-based customer segmentation.

## 📊 Dataset Description

The dataset contains **30,002 records** with the following features:

| Column | Description | Type |
|--------|-------------|------|
| `First Name` | Customer's first name | String |
| `Last Name` | Customer's last name | String |
| `Country` | Customer's country | String |
| `Car Brand` | Manufacturer of the car | String |
| `Car Model` | Specific model of the car | String |
| `Car Color` | Color of the car | String |
| `Year of Manufacture` | Year the car was manufactured | Integer |
| `Credit Card Type` | Type of credit card used | String |

### 📈 Dataset Statistics
- **Countries**: 195 unique countries
- **Car Brands**: 37 different manufacturers
- **Car Colors**: 36 distinct colors
- **Credit Card Types**: 16 different types
- **Year Range**: 1964 - 2013
- **Missing Values**: None (complete dataset)

## 🎯 Key Research Question

**Is there a correlation between the choice of car (brand, model, color) and the individual's financial profile (credit card type)?**

This analysis employs chi-square tests, Cramér's V statistics, and other statistical methods to investigate potential relationships between vehicle preferences and financial behavior patterns.

## 📁 Project Structure

```
EDA-on-Cars-Dataset/
├── 📄 cars.csv                 # Original dataset
├── 📄 codes.py                 # Legacy analysis code
├── 📄 requirements.txt         # Python dependencies
├── 📄 config.py               # Configuration settings
├── 📄 data_loader.py          # Data loading and validation
├── 📄 statistical_analysis.py # Statistical tests and analysis
├── 📄 visualizations.py       # Visualization classes
├── 📄 clustering_analysis.py  # ML clustering analysis
├── 📄 main_analysis.py        # Main analysis pipeline
└── 📄 README.md               # This file
```

## ✨ Features

### 🔬 Statistical Analysis
- **Chi-square tests** for independence between categorical variables
- **Cramér's V** effect size calculations
- **ANOVA tests** for numerical-categorical relationships
- Comprehensive descriptive statistics
- Market share analysis
- Credit card penetration analysis

### 📊 Visualizations
- **Interactive plots** using Plotly
- **Animated visualizations** showing trends over time
- **Geographic mapping** with GeoPandas
- **Correlation heatmaps**
- **Distribution plots** for all key variables
- **Silhouette analysis** for clustering validation

### 🗺️ Geographic Analysis
- World maps showing car brand preferences by country
- Color preference mapping
- Regional pattern identification

### 🤖 Machine Learning
- **K-means clustering** for customer segmentation
- **PCA visualization** for dimensionality reduction
- **Silhouette analysis** for cluster validation
- **Elbow method** for optimal cluster selection

### 🛠️ Data Quality
- Automated data validation
- Missing value detection and handling
- Data type verification
- Outlier identification

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
```bash
git clone <repository-url>
cd EDA-on-Cars-Dataset
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### 📦 Dependencies
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Static plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive visualizations
- `geopandas` - Geographic data analysis
- `scikit-learn` - Machine learning algorithms
- `scipy` - Statistical functions
- `jupyter` - Notebook environment

## 🚀 Usage

### Quick Start

Run the complete analysis pipeline:

```python
python main_analysis.py
```

### Module-by-Module Usage

#### 1. Data Loading and Validation
```python
from data_loader import DataLoader

loader = DataLoader('cars.csv')
df = loader.load_data()
is_valid, issues = loader.validate_data()
df_clean = loader.clean_data()
```

#### 2. Statistical Analysis
```python
from statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(df_clean)
correlation_results = analyzer.analyze_car_credit_correlation()
market_share = analyzer.brand_market_share_analysis()
```

#### 3. Visualizations
```python
from visualizations import CarDataVisualizer, GeographicVisualizer

# Basic visualizations
visualizer = CarDataVisualizer(df_clean)
visualizer.plot_year_distribution()
visualizer.plot_top_brands()
visualizer.plot_correlation_heatmap()

# Geographic visualizations
geo_viz = GeographicVisualizer(df_clean)
geo_viz.plot_brands_by_country()
```

#### 4. Customer Segmentation
```python
from clustering_analysis import CustomerSegmentation

segmentation = CustomerSegmentation(df_clean)
encoded_data = segmentation.prepare_data_for_clustering()
optimal_k = segmentation.find_optimal_clusters()
results = segmentation.perform_clustering()
segmentation.visualize_clusters()
```

### 📓 Jupyter Notebook Usage

For interactive analysis, you can also use the modules in Jupyter notebooks:

```bash
jupyter notebook
```

## 📊 Analysis Results

### 🔍 Key Findings

#### Statistical Correlations
- **Car Brand vs Credit Card Type**: [Significant/Not Significant]
- **Car Color vs Credit Card Type**: [Significant/Not Significant]
- **Year of Manufacture vs Credit Card Type**: [Significant/Not Significant]
- **Country vs Credit Card Type**: [Significant/Not Significant]

#### Market Share Insights
- **Top Car Brand**: Ford (X.X% market share)
- **Market Concentration**: Top 3 brands control XX% of the market
- **Geographic Distribution**: XX countries represented

#### Customer Segmentation
- **Optimal Clusters**: X distinct customer segments identified
- **Clustering Quality**: Silhouette score of X.XX
- **Segment Characteristics**: [Brief description of each segment]

### 📈 Temporal Trends
- **Peak Manufacturing Period**: 1990s-2000s
- **Brand Evolution**: Historical popularity shifts
- **Geographic Patterns**: Regional preferences

## 🔧 Improvements Made

### 🏗️ Code Architecture
- **Modular Design**: Separated concerns into dedicated modules
- **Object-Oriented Approach**: Used classes for better organization
- **Error Handling**: Added comprehensive error checking
- **Type Hints**: Improved code readability and IDE support
- **Configuration Management**: Centralized settings in config.py

### 📊 Enhanced Analysis
- **Statistical Rigor**: Added proper statistical testing
- **Effect Size Measures**: Included Cramér's V for correlation strength
- **Missing Value Handling**: Robust data cleaning pipeline
- **Data Validation**: Automated quality checks
- **Reproducibility**: Fixed random states and versioned dependencies

### 🎨 Visualization Improvements
- **Interactive Elements**: Added Plotly-based interactive charts
- **Geographic Mapping**: Implemented world maps with GeoPandas
- **Animation**: Created time-series animations
- **Professional Styling**: Consistent color schemes and formatting
- **Export Capability**: Added plot saving functionality

### 🤖 Machine Learning Enhancements
- **Automated Cluster Selection**: Implemented elbow method and silhouette analysis
- **Dimensionality Reduction**: Added PCA for visualization
- **Model Validation**: Comprehensive clustering evaluation
- **Scalability**: Efficient data preprocessing pipeline

### 📚 Documentation
- **Comprehensive README**: Detailed project documentation
- **Code Comments**: Extensive inline documentation
- **Docstrings**: Professional function documentation
- **Usage Examples**: Clear usage instructions

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### 📋 Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Project Maintainer**: [Krishna Kamal]
- **Email**: [ht785618@gmail.com]


## 🙏 Acknowledgments

- Dataset source and original analysis inspiration
- Contributors and reviewers
- Open-source libraries and tools used

---

**⭐ If you find this project helpful, please consider giving it a star!** 