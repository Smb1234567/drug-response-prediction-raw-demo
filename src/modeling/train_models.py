"""
ML Model Training for Drug Response Prediction (Dual Output + SHAP)
File: src/modeling/train_models.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

import warnings
warnings.filterwarnings('ignore')


class DrugResponseModelTrainer:
    def __init__(self, data_dir='data/processed', models_dir='models'):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}
    
    def load_data(self):
        print("=== Loading Processed Data ===")
        self.X_train = pd.read_csv(self.data_dir / 'X_train.csv')
        self.X_test = pd.read_csv(self.data_dir / 'X_test.csv')
        self.y_train_class = pd.read_csv(self.data_dir / 'y_class_train.csv').values.ravel()
        self.y_test_class = pd.read_csv(self.data_dir / 'y_class_test.csv').values.ravel()
        self.y_train_reg = pd.read_csv(self.data_dir / 'y_reg_train.csv').values.ravel()
        self.y_test_reg = pd.read_csv(self.data_dir / 'y_reg_test.csv').values.ravel()
        
        print(f"✓ Training samples: {len(self.X_train)}")
        print(f"✓ Features: {self.X_train.shape[1]}")
        
        # Clean data
        for df in [self.X_train, self.X_test]:
            df.fillna(0, inplace=True)
            df.replace([np.inf, -np.inf], 0, inplace=True)
        print("✓ Data validation complete")
        return self
    
    def define_models(self):
        print("\n=== Defining Models (Classification + Regression) ===")
        
        # Classification models
        self.class_models = {
            'XGBoost_Class': XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM_Class': LGBMClassifier(
                n_estimators=200,
                max_depth=7,
                num_leaves=127,  # ✅ Fixed: 2^7 - 1 = 127
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=1  # 🔍 Keep logs for transparency (1 = show progress)
            ),
            'RandomForest_Class': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Regression models
        self.reg_models = {
            'XGBoost_Reg': XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM_Reg': LGBMRegressor(
                n_estimators=200,
                max_depth=7,
                num_leaves=127,  # ✅ Fixed: 2^7 - 1 = 127
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=1  # 🔍 Keep logs for transparency
            ),
            'RandomForest_Reg': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        }
        
        print(f"✓ Defined {len(self.class_models)} classification and {len(self.reg_models)} regression models")
        return self
    def train_and_evaluate_classification(self):
        print("\n=== Training Classification Models ===")
        self.class_results = {}
        
        for name, model in self.class_models.items():
            print(f"\n--- Training {name} ---")
            start_time = datetime.now()
            
            model.fit(self.X_train, self.y_train_class)
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(self.y_test_class, y_pred),
                'precision': precision_score(self.y_test_class, y_pred, zero_division=0),
                'recall': recall_score(self.y_test_class, y_pred, zero_division=0),
                'f1_score': f1_score(self.y_test_class, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(self.y_test_class, y_pred_proba),
                'training_time': (datetime.now() - start_time).total_seconds()
            }
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
            joblib.dump(model, self.models_dir / f"{name.lower()}_model.pkl")
            self.class_results[name] = metrics
        
        return self
    
    def train_and_evaluate_regression(self):
        print("\n=== Training Regression Models ===")
        self.reg_results = {}
        
        for name, model in self.reg_models.items():
            print(f"\n--- Training {name} ---")
            start_time = datetime.now()
            
            model.fit(self.X_train, self.y_train_reg)
            y_pred = model.predict(self.X_test)
            
            metrics = {
                'mse': mean_squared_error(self.y_test_reg, y_pred),
                'r2': r2_score(self.y_test_reg, y_pred),
                'training_time': (datetime.now() - start_time).total_seconds()
            }
            
            print(f"  R²: {metrics['r2']:.4f}, MSE: {metrics['mse']:.4f}")
            joblib.dump(model, self.models_dir / f"{name.lower()}_model.pkl")
            self.reg_results[name] = metrics
        
        return self
    
    def compare_models(self):
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Classification comparison
        class_df = pd.DataFrame({
            'Model': list(self.class_results.keys()),
            'Accuracy': [r['accuracy'] for r in self.class_results.values()],
            'ROC-AUC': [r['roc_auc'] for r in self.class_results.values()],
            'Time(s)': [r['training_time'] for r in self.class_results.values()]
        }).sort_values('Accuracy', ascending=False)
        
        print("\nClassification Models:")
        print(class_df.to_string(index=False))
        class_df.to_csv(self.models_dir / 'classification_comparison.csv', index=False)
        
        # Regression comparison
        reg_df = pd.DataFrame({
            'Model': list(self.reg_results.keys()),
            'R2': [r['r2'] for r in self.reg_results.values()],
            'MSE': [r['mse'] for r in self.reg_results.values()],
            'Time(s)': [r['training_time'] for r in self.reg_results.values()]
        }).sort_values('R2', ascending=False)
        
        print("\nRegression Models:")
        print(reg_df.to_string(index=False))
        reg_df.to_csv(self.models_dir / 'regression_comparison.csv', index=False)
        
        # Save best models info
        best_class = class_df.iloc[0]['Model']
        best_reg = reg_df.iloc[0]['Model']
        
        best_model_info = {
            'best_classification_model': best_class,
            'best_classification_accuracy': float(class_df.iloc[0]['Accuracy']),
            'best_regression_model': best_reg,
            'best_regression_r2': float(reg_df.iloc[0]['R2']),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.models_dir / 'best_model_info.json', 'w') as f:
            json.dump(best_model_info, f, indent=4)
        
        print(f"\n🏆 Best Classification: {best_class}")
        print(f"🏆 Best Regression: {best_reg}")
        return best_class, best_reg
    
    def generate_shap_explainer(self, best_class_model_name):
        print("\n" + "="*60)
        print("GENERATING SHAP EXPLAINER")
        print("="*60)
        
        try:
            import shap
            print("✓ SHAP library available")
            
            model_path = self.models_dir / f"{best_class_model_name.lower()}_model.pkl"
            best_model = joblib.load(model_path)
            
            # Use subset of training data as background
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            
            explainer = shap.TreeExplainer(
                best_model,
                background,
                feature_names=self.X_train.columns.tolist()
            )
            
            explainer_path = self.models_dir / "shap_explainer.pkl"
            joblib.dump(explainer, explainer_path)
            print(f"✓ SHAP explainer saved to {explainer_path}")
            
        except Exception as e:
            print(f"⚠ Failed to generate SHAP explainer: {str(e)}")
    
    def run_pipeline(self):
        self.load_data()
        self.define_models()
        self.train_and_evaluate_classification()
        self.train_and_evaluate_regression()
        best_class, best_reg = self.compare_models()
        self.generate_shap_explainer(best_class)
        print("\n" + "="*60)
        print("✓ MODEL TRAINING COMPLETE!")
        print(f"✓ Models saved in: {self.models_dir}")
        print("="*60)


def main():
    trainer = DrugResponseModelTrainer()
    trainer.run_pipeline()


if __name__ == "__main__":
    main()
