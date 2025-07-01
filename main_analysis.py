"""
Main analysis script for the Cars EDA project.
This script demonstrates how to use all the modules together for comprehensive analysis.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_loader import DataLoader
from statistical_analysis import StatisticalAnalyzer
from visualizations import CarDataVisualizer, GeographicVisualizer
from clustering_analysis import CustomerSegmentation

def main():
    """
    Execute the complete EDA pipeline.
    """
    print("="*80)
    print("COMPREHENSIVE CARS DATASET ANALYSIS")
    print("="*80)
    
    # 1. Load and validate data
    print("\n1. LOADING AND VALIDATING DATA")
    print("-" * 40)
    
    loader = DataLoader()
    df = loader.load_data()
    
    # Validate data
    is_valid, issues = loader.validate_data()
    if not is_valid:
        print(f"Data validation issues found: {issues}")
    else:
        print("✓ Data validation passed")
    
    # Get data summary
    summary = loader.get_data_summary()
    print(f"Dataset shape: {summary['shape']}")
    print(f"Memory usage: {summary['memory_usage'] / 1024**2:.2f} MB")
    
    # Clean data
    df_clean = loader.clean_data()
    print(f"✓ Data cleaned. Final shape: {df_clean.shape}")
    
    # 2. Statistical Analysis
    print("\n2. STATISTICAL ANALYSIS")
    print("-" * 40)
    
    analyzer = StatisticalAnalyzer(df_clean)
    
    # Get descriptive statistics
    desc_stats = analyzer.get_descriptive_statistics()
    print("✓ Descriptive statistics calculated")
    
    # Analyze correlation between car choices and credit card types
    print("\nAnalyzing correlation between car choices and credit card types...")
    correlation_results = analyzer.analyze_car_credit_correlation()
    
    for analysis_name, result in correlation_results.items():
        print(f"\n{analysis_name.replace('_', ' ').title()}:")
        print(f"  Chi-square statistic: {result['chi2_statistic']:.4f}")
        print(f"  P-value: {result['p_value']:.4f}")
        print(f"  Cramér's V: {result['cramers_v']:.4f}")
        print(f"  Interpretation: {result['interpretation']}")
    
    # Market share analysis
    market_share = analyzer.brand_market_share_analysis()
    print(f"\n✓ Market share analysis completed")
    print("Top 5 brands by market share:")
    print(market_share.head())
    
    # 3. Data Visualization
    print("\n3. DATA VISUALIZATION")
    print("-" * 40)
    
    visualizer = CarDataVisualizer(df_clean)
    
    print("Creating visualizations...")
    
    # Basic distributions
    print("  → Year distribution")
    visualizer.plot_year_distribution()
    
    print("  → Top car brands")
    visualizer.plot_top_brands(top_n=15)
    
    print("  → Car colors distribution")
    visualizer.plot_car_colors()
    
    print("  → Credit card types distribution")
    visualizer.plot_credit_cards()
    
    print("  → Correlation heatmap")
    visualizer.plot_correlation_heatmap()
    
    # Interactive visualizations
    print("  → Interactive brand visualization")
    visualizer.plot_brands_interactive()
    
    print("  → Animated yearly trends")
    visualizer.plot_yearly_brand_animation()
    
    # 4. Geographic Analysis
    print("\n4. GEOGRAPHIC ANALYSIS")
    print("-" * 40)
    
    geo_visualizer = GeographicVisualizer(df_clean)
    
    print("Creating geographic visualizations...")
    print("  → Car brands by country")
    geo_visualizer.plot_brands_by_country()
    
    print("  → Car colors by country")
    geo_visualizer.plot_colors_by_country()
    
    # 5. Customer Segmentation
    print("\n5. CUSTOMER SEGMENTATION ANALYSIS")
    print("-" * 40)
    
    segmentation = CustomerSegmentation(df_clean)
    
    # Prepare data for clustering
    print("Preparing data for clustering...")
    features = ['Car Brand', 'Country', 'Car Color', 'Credit Card Type', 'Year of Manufacture']
    encoded_data = segmentation.prepare_data_for_clustering(features)
    print(f"✓ Data prepared. Shape: {encoded_data.shape}")
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    optimal_k = segmentation.find_optimal_clusters(max_k=10, method='silhouette')
    
    # Perform clustering
    print(f"Performing clustering with {optimal_k} clusters...")
    clustering_results = segmentation.perform_clustering(n_clusters=optimal_k)
    
    print(f"✓ Clustering completed")
    print(f"  Silhouette score: {clustering_results['silhouette_score']:.3f}")
    
    # Visualize clusters
    print("Creating cluster visualizations...")
    segmentation.visualize_clusters()
    segmentation.plot_silhouette_analysis()
    
    # Get cluster summary
    cluster_summary = segmentation.get_cluster_summary()
    print("\nCluster Summary:")
    print(cluster_summary.to_string(index=False))
    
    # 6. Key Findings and Insights
    print("\n6. KEY FINDINGS AND INSIGHTS")
    print("-" * 40)
    
    print_key_insights(correlation_results, market_share, clustering_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)

def print_key_insights(correlation_results, market_share, clustering_results):
    """
    Print key insights from the analysis.
    
    Args:
        correlation_results: Results from correlation analysis
        market_share: Market share analysis results
        clustering_results: Clustering analysis results
    """
    print("\n🔍 KEY INSIGHTS:")
    
    # Market insights
    top_brand = market_share.iloc[0]
    print(f"\n📊 MARKET INSIGHTS:")
    print(f"  • Top car brand: {top_brand['Brand']} ({top_brand['Market_Share_Percent']}% market share)")
    print(f"  • Top 3 brands control {market_share.head(3)['Market_Share_Percent'].sum():.1f}% of the market")
    
    # Correlation insights
    print(f"\n🔗 CORRELATION INSIGHTS:")
    significant_correlations = []
    for analysis_name, result in correlation_results.items():
        if result['is_significant']:
            significant_correlations.append(analysis_name)
            print(f"  • {analysis_name.replace('_', ' ').title()}: {result['interpretation']}")
    
    if not significant_correlations:
        print("  • No statistically significant correlations found between car choices and credit card types")
    
    # Clustering insights
    print(f"\n👥 CUSTOMER SEGMENTATION INSIGHTS:")
    print(f"  • Identified {clustering_results['n_clusters']} distinct customer segments")
    print(f"  • Clustering quality (silhouette score): {clustering_results['silhouette_score']:.3f}")
    
    cluster_analysis = clustering_results['cluster_analysis']
    largest_cluster = max(cluster_analysis.keys(), 
                         key=lambda x: cluster_analysis[x]['size'])
    print(f"  • Largest segment: {largest_cluster} ({cluster_analysis[largest_cluster]['percentage']:.1f}% of customers)")

if __name__ == "__main__":
    main() 