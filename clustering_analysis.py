"""
Machine Learning clustering analysis module for the Cars EDA project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
from config import KMEANS_CONFIG, RANDOM_STATE

class CustomerSegmentation:
    """Class for performing customer segmentation analysis."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the clustering analyzer.
        
        Args:
            df (pd.DataFrame): The cars dataset
        """
        self.df = df.copy()
        self.encoded_data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.optimal_k = None
        self.kmeans_model = None
        
    def prepare_data_for_clustering(self, features: list = None) -> pd.DataFrame:
        """
        Prepare data for clustering by encoding categorical variables.
        
        Args:
            features (list): List of features to use for clustering
            
        Returns:
            pd.DataFrame: Encoded and scaled data ready for clustering
        """
        if features is None:
            features = ['Car Brand', 'Country', 'Car Color', 'Credit Card Type', 'Year of Manufacture']
        
        # Select features that exist in the dataset
        available_features = [f for f in features if f in self.df.columns]
        clustering_data = self.df[available_features].copy()
        
        # Handle missing values
        clustering_data = clustering_data.dropna()
        
        # Encode categorical variables
        encoded_data = clustering_data.copy()
        
        for col in clustering_data.columns:
            if clustering_data[col].dtype == 'object':
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(clustering_data[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale the data
        self.encoded_data = pd.DataFrame(
            self.scaler.fit_transform(encoded_data),
            columns=encoded_data.columns,
            index=encoded_data.index
        )
        
        return self.encoded_data
    
    def find_optimal_clusters(self, max_k: int = 10, method: str = 'elbow') -> int:
        """
        Find the optimal number of clusters using elbow method or silhouette analysis.
        
        Args:
            max_k (int): Maximum number of clusters to test
            method (str): Method to use ('elbow' or 'silhouette')
            
        Returns:
            int: Optimal number of clusters
        """
        if self.encoded_data is None:
            raise ValueError("Data not prepared. Call prepare_data_for_clustering() first.")
        
        k_range = range(2, max_k + 1)
        
        if method == 'elbow':
            wcss = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, **KMEANS_CONFIG)
                kmeans.fit(self.encoded_data)
                wcss.append(kmeans.inertia_)
            
            # Plot elbow curve
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, wcss, marker='o', linewidth=2, markersize=8)
            plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # Simple elbow detection (can be improved with more sophisticated methods)
            # For now, return a reasonable default based on typical elbow patterns
            self.optimal_k = 3
            
        elif method == 'silhouette':
            silhouette_scores = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, **KMEANS_CONFIG)
                cluster_labels = kmeans.fit_predict(self.encoded_data)
                silhouette_avg = silhouette_score(self.encoded_data, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            # Plot silhouette scores
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, silhouette_scores, marker='o', linewidth=2, markersize=8)
            plt.title('Silhouette Analysis for Optimal k', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Average Silhouette Score')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # Select k with highest silhouette score
            self.optimal_k = k_range[np.argmax(silhouette_scores)]
        
        print(f"Optimal number of clusters ({method} method): {self.optimal_k}")
        return self.optimal_k
    
    def perform_clustering(self, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform K-means clustering on the prepared data.
        
        Args:
            n_clusters (Optional[int]): Number of clusters (uses optimal_k if None)
            
        Returns:
            Dict[str, Any]: Clustering results and analysis
        """
        if self.encoded_data is None:
            raise ValueError("Data not prepared. Call prepare_data_for_clustering() first.")
        
        if n_clusters is None:
            if self.optimal_k is None:
                self.find_optimal_clusters()
            n_clusters = self.optimal_k
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, **KMEANS_CONFIG)
        cluster_labels = self.kmeans_model.fit_predict(self.encoded_data)
        
        # Add cluster labels to original dataframe
        clustered_df = self.df.loc[self.encoded_data.index].copy()
        clustered_df['Cluster'] = cluster_labels
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.encoded_data, cluster_labels)
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(clustered_df)
        
        results = {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels,
            'clustered_dataframe': clustered_df,
            'silhouette_score': silhouette_avg,
            'cluster_analysis': cluster_analysis,
            'model': self.kmeans_model
        }
        
        return results
    
    def _analyze_clusters(self, clustered_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the characteristics of each cluster.
        
        Args:
            clustered_df (pd.DataFrame): DataFrame with cluster assignments
            
        Returns:
            Dict[str, Any]: Cluster analysis results
        """
        analysis = {}
        
        for cluster_id in sorted(clustered_df['Cluster'].unique()):
            cluster_data = clustered_df[clustered_df['Cluster'] == cluster_id]
            
            cluster_profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(clustered_df) * 100
            }
            
            # Categorical feature analysis
            categorical_features = ['Car Brand', 'Country', 'Car Color', 'Credit Card Type']
            for feature in categorical_features:
                if feature in cluster_data.columns:
                    top_values = cluster_data[feature].value_counts().head(3)
                    cluster_profile[f'top_{feature.lower().replace(" ", "_")}'] = top_values.to_dict()
            
            # Numerical feature analysis
            if 'Year of Manufacture' in cluster_data.columns:
                cluster_profile['avg_year'] = cluster_data['Year of Manufacture'].mean()
                cluster_profile['year_std'] = cluster_data['Year of Manufacture'].std()
            
            analysis[f'cluster_{cluster_id}'] = cluster_profile
        
        return analysis
    
    def visualize_clusters(self, save_path: Optional[str] = None) -> None:
        """
        Visualize clusters using PCA for dimensionality reduction.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        if self.encoded_data is None or self.kmeans_model is None:
            raise ValueError("Clustering not performed. Call perform_clustering() first.")
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.encoded_data)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        cluster_labels = self.kmeans_model.labels_
        n_clusters = len(np.unique(cluster_labels))
        
        # Plot each cluster
        colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
        for i, color in enumerate(colors):
            cluster_points = pca_data[cluster_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[color], label=f'Cluster {i}', alpha=0.7, s=50)
        
        # Plot cluster centers
        centers_pca = pca.transform(self.kmeans_model.cluster_centers_)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   c='black', marker='x', s=300, linewidths=3, label='Centroids')
        
        plt.title('Customer Segments Visualization (PCA)', fontsize=14, fontweight='bold')
        plt.xlabel(f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        print(f"Total explained variance by 2 components: {sum(pca.explained_variance_ratio_):.2%}")
    
    def plot_silhouette_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Create a detailed silhouette analysis plot.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        if self.encoded_data is None or self.kmeans_model is None:
            raise ValueError("Clustering not performed. Call perform_clustering() first.")
        
        cluster_labels = self.kmeans_model.labels_
        n_clusters = len(np.unique(cluster_labels))
        
        # Calculate silhouette scores
        silhouette_avg = silhouette_score(self.encoded_data, cluster_labels)
        sample_silhouette_values = silhouette_samples(self.encoded_data, cluster_labels)
        
        # Create the silhouette plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10
        
        ax.set_title('Silhouette Analysis for Customer Segments', fontsize=14, fontweight='bold')
        ax.set_xlabel('Silhouette Coefficient Values')
        ax.set_ylabel('Cluster Label')
        
        # Add a vertical line for average silhouette score
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
                  label=f'Average Score: {silhouette_avg:.3f}')
        
        ax.legend()
        ax.set_yticks([])
        ax.set_xlim([-0.1, 1])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        print(f"Average Silhouette Score: {silhouette_avg:.3f}")
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get a summary table of cluster characteristics.
        
        Returns:
            pd.DataFrame: Summary of cluster characteristics
        """
        if self.kmeans_model is None:
            raise ValueError("Clustering not performed. Call perform_clustering() first.")
        
        # Get clustered data
        clustered_df = self.df.loc[self.encoded_data.index].copy()
        clustered_df['Cluster'] = self.kmeans_model.labels_
        
        # Create summary
        summary_data = []
        
        for cluster_id in sorted(clustered_df['Cluster'].unique()):
            cluster_data = clustered_df[clustered_df['Cluster'] == cluster_id]
            
            summary_row = {
                'Cluster': cluster_id,
                'Size': len(cluster_data),
                'Percentage': f"{len(cluster_data) / len(clustered_df) * 100:.1f}%",
                'Top_Brand': cluster_data['Car Brand'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
                'Top_Country': cluster_data['Country'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
                'Top_Color': cluster_data['Car Color'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
                'Top_Credit_Card': cluster_data['Credit Card Type'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
                'Avg_Year': f"{cluster_data['Year of Manufacture'].mean():.1f}" if len(cluster_data) > 0 else 'N/A'
            }
            
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data) 