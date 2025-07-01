"""
Data loading and validation module for the Cars EDA project.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging
from config import DATA_FILE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and validating the cars dataset."""
    
    def __init__(self, file_path: str = DATA_FILE):
        """
        Initialize the DataLoader.
        
        Args:
            file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the cars dataset from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If the data file is not found
            Exception: If there's an error reading the file
        """
        try:
            logger.info(f"Loading data from {self.file_path}")
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self) -> Tuple[bool, list]:
        """
        Validate the loaded dataset.
        
        Returns:
            Tuple[bool, list]: (is_valid, list_of_issues)
        """
        if self.df is None:
            return False, ["No data loaded"]
        
        issues = []
        expected_columns = [
            'First Name', 'Last Name', 'Country', 'Car Brand', 
            'Car Model', 'Car Color', 'Year of Manufacture', 'Credit Card Type'
        ]
        
        # Check if all expected columns exist
        missing_columns = set(expected_columns) - set(self.df.columns)
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        # Check for empty dataset
        if self.df.empty:
            issues.append("Dataset is empty")
        
        # Check data types
        if 'Year of Manufacture' in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df['Year of Manufacture']):
                issues.append("Year of Manufacture should be numeric")
        
        # Check for reasonable year range
        if 'Year of Manufacture' in self.df.columns:
            year_range = self.df['Year of Manufacture'].dropna()
            if not year_range.empty:
                if year_range.min() < 1800 or year_range.max() > 2030:
                    issues.append("Year of Manufacture contains unrealistic values")
        
        is_valid = len(issues) == 0
        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation issues: {issues}")
        
        return is_valid, issues
    
    def get_data_summary(self) -> dict:
        """
        Get a summary of the dataset.
        
        Returns:
            dict: Summary statistics and information
        """
        if self.df is None:
            return {}
        
        summary = {
            'shape': self.df.shape,
            'null_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'unique_values': {col: self.df[col].nunique() for col in self.df.columns},
            'memory_usage': self.df.memory_usage(deep=True).sum(),
        }
        
        return summary
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and data type issues.
        
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df_cleaned = self.df.copy()
        
        # Remove rows with missing critical information
        critical_columns = ['Car Brand', 'Country', 'Year of Manufacture']
        df_cleaned = df_cleaned.dropna(subset=critical_columns)
        
        # Ensure Year of Manufacture is integer
        if 'Year of Manufacture' in df_cleaned.columns:
            df_cleaned['Year of Manufacture'] = pd.to_numeric(
                df_cleaned['Year of Manufacture'], errors='coerce'
            ).astype('Int64')
        
        # Strip whitespace from string columns
        string_columns = df_cleaned.select_dtypes(include=['object']).columns
        for col in string_columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
        
        logger.info(f"Data cleaned. Shape changed from {self.df.shape} to {df_cleaned.shape}")
        
        return df_cleaned 