"""
ML Model Training for Drug Response Prediction
File: src/modeling/train_models.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')


class DrugResponseModelTrainer:
    """Train and evaluate multiple ML models"""
    
    def __init__(self, data_dir='data/processed', models_dir='models'):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """Load preprocessed training data"""
        print("=== Loading Processed Data ===")
        
        self.X_train = pd.read_csv(self.data_dir / 'X_train.csv')
        self.X_test = pd.read_csv(self.data_dir / 'X_test.csv')
        self.y_train = pd.read_csv(self.data_dir / 'y_class_train.csv').values.ravel()
        self.y_test = pd.read_csv(self.data_dir / 'y_class_test.csv').values.ravel()
        
        # Data validation
        print(f"✓ Training samples: {len(self.X_train)}")
        print(f"✓ Test samples: {len(self.X_test)}")
        print(f"✓ Features: {self.X_train.shape[1]}")
        print(f"✓ Class distribution (train): {np.bincount(self.y_train)}")
        
        # Check for NaN or inf values
        if self.X_train.isnull().any().any():
            print("⚠ WARNING: NaN values found in training data. Cleaning...")
            self.X_train = self.X_train.fillna(0)
        
        if self.X_test.isnull().any().any():
            print("⚠ WARNING: NaN values found in test data. Cleaning...")
            self.X_test = self.X_test.fillna(0)
        
        # Replace inf with large numbers
        self.X_train = self.X_train.replace([np.inf, -np.inf], 0)
        self.X_test = self.X_test.replace([np.inf, -np.inf], 0)
        
        print("✓ Data validation complete - no NaN or inf values")
        
        return self
    
    def define_models(self):
        """Define ML models to train"""
        print("\n=== Defining Models ===")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }
        
        print(f"✓ Defined {len(self.models)} models for training")
        return self
    
    def train_model(self, name, model):
        """Train a single model and evaluate"""
        print(f"\n--- Training {name} ---")
        
        start_time = datetime.now()
        
        # Train
        model.fit(self.X_train, self.y_train)
        
        # Predict
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'training_time': (datetime.now() - start_time).total_seconds()
        }
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Time:      {metrics['training_time']:.2f}s")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_features_idx = np.argsort(importances)[-10:][::-1]
            top_features = [(self.X_train.columns[i], importances[i]) 
                           for i in top_features_idx]
            metrics['top_features'] = top_features
        
        self.results[name] = metrics
        
        # Save model
        model_path = self.models_dir / f"{name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_path)
        print(f"  ✓ Model saved: {model_path}")
        
        return model, metrics
    
    def train_all_models(self):
        """Train all defined models"""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            trained_model, metrics = self.train_model(name, model)
        
        return self
    
    def compare_models(self):
        """Compare performance of all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'F1-Score': [r['f1_score'] for r in self.results.values()],
            'ROC-AUC': [r['roc_auc'] for r in self.results.values()],
            'Time(s)': [r['training_time'] for r in self.results.values()]
        })
        
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        print("\n" + comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv(self.models_dir / 'model_comparison.csv', index=False)
        
        # Find best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_accuracy = comparison_df.iloc[0]['Accuracy']
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   Accuracy: {best_accuracy:.4f}")
        
        # Save best model info
        best_model_info = {
            'best_model': best_model_name,
            'best_accuracy': float(best_accuracy),
            'all_results': {k: {
                'accuracy': float(v['accuracy']),
                'precision': float(v['precision']),
                'recall': float(v['recall']),
                'f1_score': float(v['f1_score']),
                'roc_auc': float(v['roc_auc']),
                'training_time': float(v['training_time'])
            } for k, v in self.results.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.models_dir / 'best_model_info.json', 'w') as f:
            json.dump(best_model_info, f, indent=4)
        
        return best_model_name, comparison_df
    
    def generate_classification_report(self):
        """Generate detailed classification reports"""
        print("\n=== Generating Detailed Reports ===")
        
        for name, model_path in [(k, self.models_dir / f"{k.replace(' ', '_').lower()}_model.pkl") 
                                  for k in self.results.keys()]:
            model = joblib.load(model_path)
            y_pred = model.predict(self.X_test)
            
            report = classification_report(self.y_test, y_pred, 
                                          target_names=['Non-Responder', 'Responder'])
            
            report_path = self.models_dir / f"{name.replace(' ', '_').lower()}_report.txt"
            with open(report_path, 'w') as f:
                f.write(f"Classification Report - {name}\n")
                f.write("="*60 + "\n\n")
                f.write(report)
            
            print(f"✓ Saved report: {report_path.name}")


def main():
    """Main training pipeline"""
    trainer = DrugResponseModelTrainer()
    
    # Execute training pipeline
    trainer.load_data()
    trainer.define_models()
    trainer.train_all_models()
    trainer.compare_models()
    trainer.generate_classification_report()
    
    print("\n" + "="*60)
    print("✓ MODEL TRAINING COMPLETE!")
    print(f"✓ Models saved in: {trainer.models_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
