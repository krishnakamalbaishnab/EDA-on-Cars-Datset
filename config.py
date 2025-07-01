# Configuration settings for the EDA project

# Data file path
DATA_FILE = 'cars.csv'

# Random state for reproducibility
RANDOM_STATE = 42

# Plot settings
PLOT_STYLE = 'whitegrid'
FIGURE_SIZE = (12, 6)
LARGE_FIGURE_SIZE = (15, 10)

# Color palettes
COLOR_PALETTES = {
    'default': 'viridis',
    'car_colors': 'magma',
    'credit_cards': 'plasma',
    'brands': 'Set3'
}

# K-means clustering settings
KMEANS_CONFIG = {
    'max_iter': 300,
    'n_init': 10,
    'init': 'k-means++',
    'random_state': RANDOM_STATE
}

# Analysis settings
TOP_N_ITEMS = 10  # Number of top items to display in analysis 