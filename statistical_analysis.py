"""
Statistical analysis module for the Cars EDA project.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from scipy.stats import kruskal, f_oneway
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """Class for performing statistical analysis on the cars dataset."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the statistical analyzer.
        
        Args:
            df (pd.DataFrame): The cars dataset
        """
        self.df = df
    
    def chi_square_test(self, var1: str, var2: str) -> Dict[str, Any]:
        """
        Perform chi-square test of independence between two categorical variables.
        
        Args:
            var1 (str): First categorical variable
            var2 (str): Second categorical variable
            
        Returns:
            Dict[str, Any]: Test results including statistic, p-value, and interpretation
        """
        # Create contingency table
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate Cramér's V (effect size)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        
        # Interpret results
        alpha = 0.05
        is_significant = p_value < alpha
        
        result = {
            'test_name': 'Chi-Square Test of Independence',
            'variables': [var1, var2],
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'is_significant': is_significant,
            'alpha': alpha,
            'interpretation': self._interpret_chi_square(p_value, cramers_v, alpha),
            'contingency_table': contingency_table
        }
        
        return result
    
    def _interpret_chi_square(self, p_value: float, cramers_v: float, alpha: float) -> str:
        """
        Interpret chi-square test results.
        
        Args:
            p_value (float): P-value from the test
            cramers_v (float): Cramér's V effect size
            alpha (float): Significance level
            
        Returns:
            str: Interpretation of the results
        """
        if p_value < alpha:
            if cramers_v < 0.1:
                strength = "weak"
            elif cramers_v < 0.3:
                strength = "moderate"
            else:
                strength = "strong"
            
            return f"There is a statistically significant association ({strength} effect size: {cramers_v:.3f})"
        else:
            return "No statistically significant association found"
    
    def analyze_car_credit_correlation(self) -> Dict[str, Any]:
        """
        Analyze the correlation between car choices and credit card types.
        
        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        results = {}
        
        # 1. Car Brand vs Credit Card Type
        brand_credit_result = self.chi_square_test('Car Brand', 'Credit Card Type')
        results['brand_vs_credit'] = brand_credit_result
        
        # 2. Car Color vs Credit Card Type
        color_credit_result = self.chi_square_test('Car Color', 'Credit Card Type')
        results['color_vs_credit'] = color_credit_result
        
        # 3. Year of Manufacture analysis by Credit Card Type
        year_credit_result = self._analyze_year_by_credit()
        results['year_vs_credit'] = year_credit_result
        
        # 4. Country vs Credit Card Type
        country_credit_result = self.chi_square_test('Country', 'Credit Card Type')
        results['country_vs_credit'] = country_credit_result
        
        return results
    
    def _analyze_year_by_credit(self) -> Dict[str, Any]:
        """
        Analyze the relationship between manufacturing year and credit card types.
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Group manufacturing years into categories for better analysis
        self.df['Year_Category'] = pd.cut(
            self.df['Year of Manufacture'],
            bins=[1960, 1980, 1990, 2000, 2010, 2020],
            labels=['1960-1980', '1981-1990', '1991-2000', '2001-2010', '2011-2020'],
            include_lowest=True
        )
        
        # Perform chi-square test
        result = self.chi_square_test('Year_Category', 'Credit Card Type')
        
        # Add ANOVA test for numerical year vs credit card type
        credit_card_groups = []
        credit_card_labels = []
        
        for card_type in self.df['Credit Card Type'].unique():
            if pd.notna(card_type):
                years = self.df[self.df['Credit Card Type'] == card_type]['Year of Manufacture'].dropna()
                if len(years) > 0:
                    credit_card_groups.append(years)
                    credit_card_labels.append(card_type)
        
        if len(credit_card_groups) > 1:
            f_stat, p_value_anova = f_oneway(*credit_card_groups)
            result['anova_f_statistic'] = f_stat
            result['anova_p_value'] = p_value_anova
            result['anova_significant'] = p_value_anova < 0.05
        
        return result
    
    def get_descriptive_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive descriptive statistics for the dataset.
        
        Returns:
            Dict[str, Any]: Descriptive statistics
        """
        stats = {}
        
        # Numerical variables
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            stats['numerical'] = self.df[numerical_cols].describe()
        
        # Categorical variables
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        stats['categorical'] = {}
        
        for col in categorical_cols:
            stats['categorical'][col] = {
                'unique_count': self.df[col].nunique(),
                'most_frequent': self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else None,
                'frequency_distribution': self.df[col].value_counts().head(10).to_dict()
            }
        
        # Missing values
        stats['missing_values'] = self.df.isnull().sum().to_dict()
        
        # Dataset overview
        stats['overview'] = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': self.df.duplicated().sum()
        }
        
        return stats
    
    def brand_market_share_analysis(self) -> pd.DataFrame:
        """
        Analyze market share of car brands.
        
        Returns:
            pd.DataFrame: Market share analysis
        """
        brand_counts = self.df['Car Brand'].value_counts()
        total_cars = len(self.df)
        
        market_share = pd.DataFrame({
            'Brand': brand_counts.index,
            'Count': brand_counts.values,
            'Market_Share_Percent': (brand_counts.values / total_cars * 100).round(2),
            'Cumulative_Share': (brand_counts.values / total_cars * 100).cumsum().round(2)
        })
        
        return market_share
    
    def credit_card_penetration_analysis(self) -> Dict[str, Any]:
        """
        Analyze credit card penetration by different segments.
        
        Returns:
            Dict[str, Any]: Credit card penetration analysis
        """
        analysis = {}
        
        # Overall credit card distribution
        card_distribution = self.df['Credit Card Type'].value_counts(normalize=True) * 100
        analysis['overall_distribution'] = card_distribution.round(2).to_dict()
        
        # Credit card distribution by country
        country_card = pd.crosstab(
            self.df['Country'], 
            self.df['Credit Card Type'], 
            normalize='index'
        ) * 100
        analysis['by_country'] = country_card.round(2)
        
        # Credit card distribution by car brand
        brand_card = pd.crosstab(
            self.df['Car Brand'], 
            self.df['Credit Card Type'], 
            normalize='index'
        ) * 100
        analysis['by_car_brand'] = brand_card.round(2)
        
        return analysis
    
    def temporal_analysis(self) -> Dict[str, Any]:
        """
        Analyze temporal trends in the data.
        
        Returns:
            Dict[str, Any]: Temporal analysis results
        """
        analysis = {}
        
        # Cars by decade
        self.df['Decade'] = (self.df['Year of Manufacture'] // 10) * 10
        decade_counts = self.df['Decade'].value_counts().sort_index()
        analysis['cars_by_decade'] = decade_counts.to_dict()
        
        # Brand popularity over time
        brand_year = self.df.groupby(['Year of Manufacture', 'Car Brand']).size().unstack(fill_value=0)
        analysis['brand_trends'] = brand_year
        
        # Average year by brand
        avg_year_by_brand = self.df.groupby('Car Brand')['Year of Manufacture'].mean().sort_values(ascending=False)
        analysis['avg_year_by_brand'] = avg_year_by_brand.round(1).to_dict()
        
        return analysis 