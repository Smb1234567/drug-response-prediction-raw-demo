"""
Enhanced Fix Script to Clean AND Encode Existing Data
File: fix_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

def fix_and_encode_data(data_dir='data/processed'):
    """Clean, fix, and encode categorical variables in processed data"""
    
    data_dir = Path(data_dir)
    
    print("=== Fixing and Encoding Processed Data ===")
    print(f"Directory: {data_dir}")
    
    files = ['X_train.csv', 'X_test.csv']
    
    for filename in files:
        filepath = data_dir / filename
        
        if not filepath.exists():
            print(f"⚠ File not found: {filename}")
            continue
        
        print(f"\nProcessing: {filename}")
        
        # Load data
        df = pd.read_csv(filepath)
        original_shape = df.shape
        
        # Fix NaN and Inf values
        nan_count = df.isnull().sum().sum()
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        
        print(f"  Original shape: {original_shape}")
        print(f"  NaN values: {nan_count}")
        print(f"  Inf values: {inf_count}")
        
        # Handle missing values
        if nan_count > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df[col].fillna(median_val, inplace=True)
            df = df.fillna(0)
        
        # Handle infinite values
        if inf_count > 0:
            df = df.replace([np.inf, -np.inf], 0)
        
        # Encode categorical columns (string columns)
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        if categorical_cols:
            print(f"  Encoding categorical columns: {categorical_cols}")
            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            df = df_encoded
            print(f"  New shape after encoding: {df.shape}")
        else:
            print("  No categorical columns found")
        
        # Final verification
        final_nan = df.isnull().sum().sum()
        final_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        string_cols = [col for col in df.columns if df[col].dtype == 'object']
        
        print(f"  Final - NaN: {final_nan}, Inf: {final_inf}, String cols: {len(string_cols)}")
        
        # Save fixed data
        df.to_csv(filepath, index=False)
        print(f"  ✓ Fixed and saved: {filename}")
    
    print("\n=== Data Fix and Encoding Complete ===")


if __name__ == "__main__":
    fix_and_encode_data()
