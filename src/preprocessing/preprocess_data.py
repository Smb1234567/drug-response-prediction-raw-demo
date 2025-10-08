"""
Preprocessing Pipeline for Drug Response Prediction
File: src/preprocessing/preprocess_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

class DrugResponsePreprocessor:
    def __init__(self, raw_data_path='data/raw/drug_response_realistic.csv',
                 processed_data_dir='data/processed'):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = None
        
    def load_and_prepare_data(self):
        print("=== Loading Raw Data ===")
        df = pd.read_csv(self.raw_data_path)
        print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Separate targets
        X = df.drop(['RESPONSE', 'COSMIC_ID', 'LN_IC50', 'AUC', 'SMILES'], axis=1, errors='ignore')
        y_class = df['RESPONSE']
        y_reg = df['LN_IC50']
        
        print(f"✓ Features: {X.shape[1]}")
        return X, y_class, y_reg
    
    def create_preprocessor(self, X):
        categorical_cols = []
        numerical_cols = []
        fingerprint_cols = []
        
        for col in X.columns:
            if col.startswith('DRUG_FP_'):
                fingerprint_cols.append(col)
            elif col.startswith('MUT_'):
                continue  # Already binary
            elif X[col].dtype == 'object':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        print(f"  Categorical: {len(categorical_cols)}")
        print(f"  Numerical: {len(numerical_cols)}")
        print(f"  Fingerprints: {len(fingerprint_cols)}")
        print(f"  Binary mutations: {len([c for c in X.columns if c.startswith('MUT_')])}")
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
                ('fp', 'passthrough', fingerprint_cols)
            ],
            remainder='passthrough'  # For mutation columns
        )
        return self.preprocessor
    
    def split_and_preprocess(self, X, y_class, y_reg, test_size=0.2, random_state=42):
        print("\n=== Splitting Data ===")
        X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
            X, y_class, y_reg, test_size=test_size, random_state=random_state, stratify=y_class
        )
        print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
        
        print("\n=== Creating Preprocessor ===")
        preprocessor = self.create_preprocessor(X_train)
        
        print("=== Fitting and Transforming Data ===")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Reconstruct feature names
        num_names = numerical_cols if 'numerical_cols' in locals() else []
        try:
            num_names = preprocessor.named_transformers_['num'].get_feature_names_out()
        except:
            num_names = [col for col in X_train.columns if col in numerical_cols]
            
        try:
            cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
        except:
            cat_names = []
            
        fp_names = [col for col in X_train.columns if col.startswith('DRUG_FP_')]
        mut_names = [col for col in X_train.columns if col.startswith('MUT_')]
        all_feature_names = list(num_names) + list(cat_names) + fp_names + mut_names
        
        X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names[:X_train_processed.shape[1]])
        X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names[:X_test_processed.shape[1]])
        
        return X_train_df, X_test_df, y_train_class, y_test_class, y_train_reg, y_test_reg
    
    def save_processed_data(self, X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg):
        print("\n=== Saving Processed Data ===")
        X_train.to_csv(self.processed_data_dir / 'X_train.csv', index=False)
        X_test.to_csv(self.processed_data_dir / 'X_test.csv', index=False)
        pd.DataFrame(y_train_class, columns=['RESPONSE']).to_csv(self.processed_data_dir / 'y_class_train.csv', index=False)
        pd.DataFrame(y_test_class, columns=['RESPONSE']).to_csv(self.processed_data_dir / 'y_class_test.csv', index=False)
        pd.DataFrame(y_train_reg, columns=['LN_IC50']).to_csv(self.processed_data_dir / 'y_reg_train.csv', index=False)
        pd.DataFrame(y_test_reg, columns=['LN_IC50']).to_csv(self.processed_data_dir / 'y_reg_test.csv', index=False)
        joblib.dump(self.preprocessor, self.processed_data_dir / 'preprocessor.pkl')
        print(f"✓ Saved processed data to {self.processed_data_dir}")
    
    def run_pipeline(self):
        X, y_class, y_reg = self.load_and_prepare_data()
        X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = self.split_and_preprocess(X, y_class, y_reg)
        self.save_processed_data(X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg)
        return X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg


def main():
    preprocessor = DrugResponsePreprocessor()
    preprocessor.run_pipeline()
    print("\n" + "="*60)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
