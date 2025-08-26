"""
Data loading functionality with schema inference and type detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import logging
from datetime import datetime
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class SchemaInfo:
    """Information about dataset schema and characteristics."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize schema info from DataFrame."""
        self.shape = df.shape
        self.columns = list(df.columns)
        self.dtypes = df.dtypes.to_dict()
        self.missing_values = df.isnull().sum().to_dict()
        self.missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Infer semantic types
        self.semantic_types = self._infer_semantic_types(df)
        
        # Basic statistics
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Memory usage
        self.memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
    def _infer_semantic_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Infer semantic types for columns."""
        semantic_types = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                semantic_types[col] = "empty"
                continue
                
            dtype = str(df[col].dtype)
            
            # Numeric types
            if df[col].dtype in [np.int64, np.int32, np.float64, np.float32]:
                if df[col].nunique() <= 2:
                    semantic_types[col] = "boolean_numeric"
                elif df[col].nunique() < len(df) * 0.05:  # Few unique values
                    semantic_types[col] = "categorical_numeric"
                else:
                    semantic_types[col] = "continuous_numeric"
            
            # String/object types
            elif dtype == 'object':
                # Try to detect specific patterns
                if self._is_date_column(col_data):
                    semantic_types[col] = "date_string"
                elif self._is_id_column(col, col_data):
                    semantic_types[col] = "identifier"
                elif col_data.nunique() < len(col_data) * 0.5:  # Many duplicates
                    semantic_types[col] = "categorical"
                else:
                    semantic_types[col] = "text"
            
            # DateTime types
            elif 'datetime' in dtype:
                semantic_types[col] = "datetime"
            
            # Boolean types
            elif dtype == 'bool':
                semantic_types[col] = "boolean"
            
            else:
                semantic_types[col] = "other"
                
        return semantic_types
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a string column contains date-like values."""
        sample = series.head(100)
        date_count = 0
        
        for value in sample:
            try:
                pd.to_datetime(str(value))
                date_count += 1
            except:
                pass
        
        return date_count > len(sample) * 0.8
    
    def _is_id_column(self, col_name: str, series: pd.Series) -> bool:
        """Check if column appears to be an ID column."""
        col_lower = col_name.lower()
        id_keywords = ['id', 'key', 'index', 'uuid', 'guid']
        
        # Check column name
        name_suggests_id = any(keyword in col_lower for keyword in id_keywords)
        
        # Check uniqueness
        is_unique = series.nunique() == len(series)
        
        return name_suggests_id or is_unique
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema info to dictionary."""
        return {
            "shape": self.shape,
            "columns": self.columns,
            "dtypes": {k: str(v) for k, v in self.dtypes.items()},
            "semantic_types": self.semantic_types,
            "missing_values": self.missing_values,
            "missing_percentages": {k: round(v, 2) for k, v in self.missing_percentages.items()},
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "memory_usage_mb": round(self.memory_usage_mb, 2)
        }


class DataLoader:
    """Loads and processes datasets with schema inference."""
    
    def __init__(self):
        """Initialize data loader."""
        pass
    
    def load_dataset(
        self, 
        file_path: Union[str, Path],
        sample_size: Optional[int] = None,
        optimize_dtypes: bool = True
    ) -> pd.DataFrame:
        """
        Load dataset from file with automatic type inference.
        
        Args:
            file_path: Path to dataset file
            sample_size: Limit number of rows loaded (for large datasets)
            optimize_dtypes: Optimize memory usage by downcasting types
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        logger.info(f"Loading dataset from {file_path}")
        
        # Load based on file extension
        if file_path.suffix.lower() == '.parquet':
            df = self._load_parquet(file_path, sample_size)
        elif file_path.suffix.lower() == '.csv':
            df = self._load_csv(file_path, sample_size)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = self._load_excel(file_path, sample_size)
        elif file_path.suffix.lower() == '.json':
            df = self._load_json(file_path, sample_size)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded dataset with shape {df.shape}")
        
        # Optimize data types if requested
        if optimize_dtypes:
            df = self._optimize_dtypes(df)
            
        # Clean column names
        df = self._clean_column_names(df)
        
        return df
    
    def _load_parquet(self, file_path: Path, sample_size: Optional[int]) -> pd.DataFrame:
        """Load parquet file."""
        if sample_size:
            # For large files, we might want to read in chunks
            parquet_file = pq.ParquetFile(file_path)
            total_rows = parquet_file.metadata.num_rows
            
            if total_rows > sample_size:
                logger.info(f"Sampling {sample_size} rows from {total_rows} total rows")
                # Read first batch of rows
                df = parquet_file.read_row_group(0).to_pandas()
                if len(df) > sample_size:
                    df = df.head(sample_size)
            else:
                df = pd.read_parquet(file_path)
        else:
            df = pd.read_parquet(file_path)
            
        return df
    
    def _load_csv(self, file_path: Path, sample_size: Optional[int]) -> pd.DataFrame:
        """Load CSV file with automatic type inference."""
        # First, sample to infer types
        sample_df = pd.read_csv(file_path, nrows=1000)
        
        # Infer datetime columns
        datetime_cols = []
        for col in sample_df.columns:
            if sample_df[col].dtype == 'object':
                try:
                    pd.to_datetime(sample_df[col].dropna().head(50))
                    datetime_cols.append(col)
                except:
                    pass
        
        # Load with inferred types
        df = pd.read_csv(
            file_path,
            nrows=sample_size,
            parse_dates=datetime_cols,
            low_memory=False
        )
        
        return df
    
    def _load_excel(self, file_path: Path, sample_size: Optional[int]) -> pd.DataFrame:
        """Load Excel file."""
        df = pd.read_excel(file_path)
        
        if sample_size and len(df) > sample_size:
            df = df.head(sample_size)
            
        return df
    
    def _load_json(self, file_path: Path, sample_size: Optional[int]) -> pd.DataFrame:
        """Load JSON file."""
        df = pd.read_json(file_path)
        
        if sample_size and len(df) > sample_size:
            df = df.head(sample_size)
            
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes to reduce memory usage."""
        original_memory = df.memory_usage(deep=True).sum()
        
        # Optimize integer columns
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert low-cardinality string columns to category
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < len(df) * 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        new_memory = df.memory_usage(deep=True).sum()
        reduction = (original_memory - new_memory) / original_memory * 100
        
        if reduction > 5:  # Only log if significant reduction
            logger.info(f"Optimized memory usage by {reduction:.1f}%")
            
        return df
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names for easier analysis."""
        # Convert to lowercase and replace spaces/special chars with underscores
        new_columns = []
        for col in df.columns:
            clean_col = str(col).lower().strip()
            clean_col = ''.join(c if c.isalnum() else '_' for c in clean_col)
            clean_col = '_'.join(clean_col.split('_'))  # Remove multiple underscores
            clean_col = clean_col.strip('_')  # Remove leading/trailing underscores
            new_columns.append(clean_col)
        
        df.columns = new_columns
        return df
    
    def get_schema_info(self, df: pd.DataFrame) -> SchemaInfo:
        """Get comprehensive schema information."""
        return SchemaInfo(df)
    
    def detect_issues(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect potential data quality issues."""
        issues = []
        
        # High missing value percentage
        missing_pct = df.isnull().sum() / len(df) * 100
        for col, pct in missing_pct.items():
            if pct > 50:
                issues.append({
                    "type": "high_missing_values",
                    "column": col,
                    "severity": "high" if pct > 80 else "medium",
                    "description": f"Column '{col}' has {pct:.1f}% missing values"
                })
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            pct_dup = duplicates / len(df) * 100
            issues.append({
                "type": "duplicate_rows",
                "severity": "high" if pct_dup > 10 else "medium",
                "description": f"{duplicates} duplicate rows ({pct_dup:.1f}%)"
            })
        
        # Single-value columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                issues.append({
                    "type": "constant_column",
                    "column": col,
                    "severity": "medium",
                    "description": f"Column '{col}' has only one unique value"
                })
        
        return issues


def get_default_loader() -> DataLoader:
    """Get default data loader instance."""
    return DataLoader()
