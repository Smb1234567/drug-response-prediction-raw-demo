"""
Preprocessing Pipeline for Drug Response Prediction
File: src/preprocessing/preprocess_data.py

This version properly handles categorical encoding using ColumnTransformer
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

class DrugResponsePreprocessor:
    """Preprocess drug response data with proper categorical handling"""
    
    def __init__(self, raw_data_path='data/raw/drug_response_realistic.csv',
                 processed_data_dir='data/processed'):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = None
        
    def load_and_prepare_data(self):
        """Load raw data and separate features/target"""
        print("=== Loading Raw Data ===")
        df = pd.read_csv(self.raw_data_path)
        print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Separate target
        if 'RESPONSE' not in df.columns:
            raise ValueError("RESPONSE column not found in data")
            
        X = df.drop(['RESPONSE', 'COSMIC_ID', 'LN_IC50', 'AUC'], axis=1, errors='ignore')
        y = df['RESPONSE']
        
        print(f"✓ Features: {X.shape[1]}, Target: RESPONSE")
        return X, y
    
    def create_preprocessor(self, X):
        """Create preprocessor with proper column handling"""
        # Identify column types
        categorical_cols = []
        numerical_cols = []
        
        for col in X.columns:
            if col.startswith('MUT_'):
                # Mutation columns are already binary 0/1
                continue
            elif X[col].dtype == 'object':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        print(f"  Categorical features: {len(categorical_cols)} {categorical_cols}")
        print(f"  Numerical features: {len(numerical_cols)} {numerical_cols}")
        print(f"  Binary mutation features: {len([c for c in X.columns if c.startswith('MUT_')])}")
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'  # Keeps mutation columns unchanged
        )
        
        return self.preprocessor
    
    def split_and_preprocess(self, X, y, test_size=0.2, random_state=42):
        """Split data and apply preprocessing"""
        print("\n=== Splitting Data ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
        
        print("\n=== Creating Preprocessor ===")
        preprocessor = self.create_preprocessor(X_train)
        
        print("=== Fitting and Transforming Data ===")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Convert to DataFrames to preserve feature names (optional but helpful)
        feature_names = (
            list(preprocessor.named_transformers_['num'].get_feature_names_out() 
                 if hasattr(preprocessor.named_transformers_['num'], 'get_feature_names_out') 
                 else numerical_cols) +
            list(preprocessor.named_transformers_['cat'].get_feature_names_out()) +
            [col for col in X_train.columns if col.startswith('MUT_')]
        )
        
        # Handle case where there might be no numerical or categorical features
        try:
            num_names = preprocessor.named_transformers_['num'].get_feature_names_out()
        except:
            num_names = [col for col in X_train.columns if col in numerical_cols]
            
        try:
            cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
        except:
            cat_names = []
            
        mut_names = [col for col in X_train.columns if col.startswith('MUT_')]
        all_feature_names = list(num_names) + list(cat_names) + mut_names
        
        X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names[:X_train_processed.shape[1]])
        X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names[:X_test_processed.shape[1]])
        
        return X_train_df, X_test_df, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data to files"""
        print("\n=== Saving Processed Data ===")
        
        X_train.to_csv(self.processed_data_dir / 'X_train.csv', index=False)
        X_test.to_csv(self.processed_data_dir / 'X_test.csv', index=False)
        pd.DataFrame(y_train, columns=['RESPONSE']).to_csv(
            self.processed_data_dir / 'y_class_train.csv', index=False)
        pd.DataFrame(y_test, columns=['RESPONSE']).to_csv(
            self.processed_data_dir / 'y_class_test.csv', index=False)
        
        # Save preprocessor for future use
        joblib.dump(self.preprocessor, self.processed_data_dir / 'preprocessor.pkl')
        
        print(f"✓ Saved processed data to {self.processed_data_dir}")
        print(f"✓ Final feature count: {X_train.shape[1]}")
    
    def run_pipeline(self):
        """Run complete preprocessing pipeline"""
        X, y = self.load_and_prepare_data()
        X_train, X_test, y_train, y_test = self.split_and_preprocess(X, y)
        self.save_processed_data(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test


def main():
    """Main execution"""
    preprocessor = DrugResponsePreprocessor()
    preprocessor.run_pipeline()
    
    print("\n" + "="*60)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python src/modeling/train_models.py")
    print("2. Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
