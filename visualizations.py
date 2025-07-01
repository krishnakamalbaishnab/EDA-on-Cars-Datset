"""
Visualization module for the Cars EDA project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import geopandas as gpd
from config import PLOT_STYLE, FIGURE_SIZE, LARGE_FIGURE_SIZE, COLOR_PALETTES

# Set the default style
sns.set_style(PLOT_STYLE)

class CarDataVisualizer:
    """Class for creating visualizations of car dataset."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the visualizer with a dataset.
        
        Args:
            df (pd.DataFrame): The cars dataset
        """
        self.df = df
        
    def plot_year_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of car manufacturing years.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        sns.histplot(
            self.df['Year of Manufacture'], 
            kde=True, 
            bins=30, 
            color='blue', 
            ax=ax
        )
        
        ax.set_title('Distribution of Year of Manufacture', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year of Manufacture')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_top_brands(self, top_n: int = 15, save_path: Optional[str] = None) -> None:
        """
        Plot the top car brands.
        
        Args:
            top_n (int): Number of top brands to display
            save_path (Optional[str]): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        brand_counts = self.df['Car Brand'].value_counts().head(top_n)
        
        sns.barplot(
            x=brand_counts.values,
            y=brand_counts.index,
            palette=COLOR_PALETTES['brands'],
            ax=ax
        )
        
        ax.set_title(f'Top {top_n} Car Brands by Frequency', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Car Brand')
        
        # Add value labels on bars
        for i, v in enumerate(brand_counts.values):
            ax.text(v + 0.1, i, str(v), va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_car_colors(self, save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of car colors.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        color_counts = self.df['Car Color'].value_counts()
        
        sns.countplot(
            y=self.df['Car Color'],
            order=color_counts.index,
            palette=COLOR_PALETTES['car_colors'],
            ax=ax
        )
        
        ax.set_title('Distribution of Car Colors', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Car Color')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_credit_cards(self, save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of credit card types.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        card_counts = self.df['Credit Card Type'].value_counts()
        
        sns.countplot(
            y=self.df['Credit Card Type'],
            order=card_counts.index,
            palette=COLOR_PALETTES['credit_cards'],
            ax=ax
        )
        
        ax.set_title('Distribution of Credit Card Types', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Credit Card Type')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_brands_interactive(self) -> None:
        """Create an interactive plot of car brands using Plotly."""
        brand_counts = self.df['Car Brand'].value_counts().reset_index()
        brand_counts.columns = ['Car Brand', 'Count']
        
        fig = px.bar(
            brand_counts.head(20),
            x='Car Brand',
            y='Count',
            color='Count',
            title='Top 20 Car Brands (Interactive)',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            showlegend=False
        )
        
        fig.show()
    
    def plot_yearly_brand_animation(self) -> None:
        """Create an animated plot showing car brand trends over years."""
        # Prepare data for animation
        year_brand_data = (
            self.df.groupby(['Year of Manufacture', 'Car Brand'])
            .size()
            .reset_index(name='Count')
        )
        
        # Filter for years with sufficient data
        year_counts = self.df['Year of Manufacture'].value_counts()
        valid_years = year_counts[year_counts >= 10].index
        year_brand_data = year_brand_data[
            year_brand_data['Year of Manufacture'].isin(valid_years)
        ]
        
        fig = px.bar(
            year_brand_data,
            x='Car Brand',
            y='Count',
            color='Count',
            animation_frame='Year of Manufacture',
            animation_group='Car Brand',
            title='Car Brand Distribution Over Years (Animated)',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=600
        )
        
        fig.show()
    
    def plot_correlation_heatmap(self, save_path: Optional[str] = None) -> None:
        """
        Plot correlation heatmap for numerical and encoded categorical variables.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        # Create encoded versions of categorical variables
        df_encoded = self.df.copy()
        
        # Encode categorical variables
        categorical_cols = ['Car Brand', 'Country', 'Car Color', 'Credit Card Type']
        for col in categorical_cols:
            if col in df_encoded.columns:
                df_encoded[f'{col}_encoded'] = pd.Categorical(df_encoded[col]).codes
        
        # Select only numerical columns for correlation
        numerical_cols = ['Year of Manufacture'] + [col for col in df_encoded.columns if col.endswith('_encoded')]
        correlation_data = df_encoded[numerical_cols]
        
        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            ax=ax,
            fmt='.2f'
        )
        
        ax.set_title('Correlation Heatmap of Car Features', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class GeographicVisualizer:
    """Class for creating geographic visualizations."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the geographic visualizer.
        
        Args:
            df (pd.DataFrame): The cars dataset
        """
        self.df = df
    
    def plot_brands_by_country(self, save_path: Optional[str] = None) -> None:
        """
        Plot the most popular car brand by country on a world map.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        try:
            # Aggregate data to find the most bought car brand per country
            most_bought_brands = (
                self.df.groupby('Country')['Car Brand']
                .apply(lambda x: x.value_counts().idxmax())
                .reset_index(name='MostBoughtBrand')
            )
            
            # Load the world map dataset
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            
            # Merge the world map with the most bought car brand data
            world_brands = world.merge(
                most_bought_brands, 
                how="left", 
                left_on="name", 
                right_on="Country"
            )
            
            # Create the plot
            fig, ax = plt.subplots(figsize=LARGE_FIGURE_SIZE)
            
            # Plot world boundaries
            world.boundary.plot(ax=ax, linewidth=0.5, color='black')
            
            # Plot the choropleth map
            world_brands.plot(
                column='MostBoughtBrand',
                ax=ax,
                legend=True,
                missing_kwds={'color': 'lightgrey', 'label': 'No Data'},
                cmap='Set3',
                legend_kwds={'loc': 'lower left', 'bbox_to_anchor': (0, 0)}
            )
            
            ax.set_title('Most Popular Car Brand by Country', fontsize=16, fontweight='bold')
            ax.set_axis_off()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating geographic visualization: {str(e)}")
            print("Please ensure geopandas is properly installed with all dependencies.")
    
    def plot_colors_by_country(self, save_path: Optional[str] = None) -> None:
        """
        Plot the most popular car color by country on a world map.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        try:
            # Aggregate data to find the most used car color per country
            most_used_colors = (
                self.df.groupby('Country')['Car Color']
                .apply(lambda x: x.value_counts().idxmax())
                .reset_index(name='MostUsedColor')
            )
            
            # Load the world map dataset
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            
            # Merge the world map with the most used car color data
            world_colors = world.merge(
                most_used_colors,
                how="left",
                left_on="name",
                right_on="Country"
            )
            
            # Create the plot
            fig, ax = plt.subplots(figsize=LARGE_FIGURE_SIZE)
            
            # Plot world boundaries
            world.boundary.plot(ax=ax, linewidth=0.5, color='black')
            
            # Plot the choropleth map
            world_colors.plot(
                column='MostUsedColor',
                ax=ax,
                legend=True,
                missing_kwds={'color': 'lightgrey', 'label': 'No Data'},
                cmap='tab20',
                legend_kwds={'loc': 'lower left', 'bbox_to_anchor': (0, 0)}
            )
            
            ax.set_title('Most Popular Car Color by Country', fontsize=16, fontweight='bold')
            ax.set_axis_off()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating geographic visualization: {str(e)}")
            print("Please ensure geopandas is properly installed with all dependencies.") 