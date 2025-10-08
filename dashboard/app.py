"""
Drug Response Prediction Dashboard – Project Heisenberg (with SHAP & Dual Output)
File: dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

SHAP_AVAILABLE = True
try:
    import shap
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="Project Heisenberg: Drug Response Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #2c3e50; text-align: center; margin-bottom: 2rem; }
    .heisenberg-tag { text-align: center; color: #e74c3c; font-weight: bold; margin-bottom: 1.5rem; font-size: 1.2rem; }
    .prediction-box { padding: 2rem; border-radius: 1rem; text-align: center; font-size: 1.5rem; font-weight: bold; }
    .responder { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .non-responder { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    .ic50-box { background-color: #e3f2fd; color: #0d47a1; border: 2px solid #bbdefb; }
    .footer-quote { font-style: italic; font-weight: bold; color: #2c3e50; text-align: center; margin-top: 2rem; padding: 1rem; border-top: 2px solid #bdc3c7; }
</style>
""", unsafe_allow_html=True)


class DrugResponsePredictor:
    def __init__(self, models_dir='models', data_dir='data/processed'):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.load_artifacts()
    
    def load_artifacts(self):
        try:
            with open(self.models_dir / 'best_model_info.json', 'r') as f:
                self.model_info = json.load(f)
            
            # Load classification model
            best_class = self.model_info['best_classification_model']
            self.class_model = joblib.load(self.models_dir / f"{best_class.lower()}_model.pkl")
            
            # Load regression model
            best_reg = self.model_info['best_regression_model']
            self.reg_model = joblib.load(self.models_dir / f"{best_reg.lower()}_model.pkl")
            
            X_test = pd.read_csv(self.data_dir / 'X_test.csv')
            self.feature_names = X_test.columns.tolist()
            
            # Load SHAP explainer
            self.explainer = None
            if SHAP_AVAILABLE:
                try:
                    self.explainer = joblib.load(self.models_dir / "shap_explainer.pkl")
                except:
                    self.explainer = None
            
            self.loaded = True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            self.loaded = False
    
    def _create_encoded_features(self, input_data):
        encoded_input = {feat: 0.0 for feat in self.feature_names}
        encoded_input['AGE'] = float(input_data['AGE'])
        encoded_input['BMI'] = float(input_data['BMI'])
        encoded_input['PRIOR_TREATMENT'] = float(input_data['PRIOR_TREATMENT'])
        
        for key, value in input_data.items():
            if key.startswith('MUT_') and key in encoded_input:
                encoded_input[key] = float(value)
        
        if input_data['GENDER'] == 'Male' and 'GENDER_Male' in encoded_input:
            encoded_input['GENDER_Male'] = 1.0
        
        stage_col = f"STAGE_{input_data['STAGE']}"
        if stage_col in encoded_input:
            encoded_input[stage_col] = 1.0
        
        tissue_col = f"TISSUE_{input_data['TISSUE']}"
        if tissue_col in encoded_input:
            encoded_input[tissue_col] = 1.0
        
        drug_col = f"DRUG_NAME_{input_data['DRUG_NAME']}"
        if drug_col in encoded_input:
            encoded_input[drug_col] = 1.0
        
        # Add dummy fingerprint features (in real system, compute from SMILES)
        for i in range(128):
            fp_col = f"DRUG_FP_{i}"
            if fp_col in encoded_input:
                encoded_input[fp_col] = 0.0  # Placeholder
        
        return encoded_input
    
    def predict_with_shap(self, input_data):
        encoded_input = self._create_encoded_features(input_data)
        input_df = pd.DataFrame([encoded_input])
        input_df = input_df[self.feature_names]
        
        # Classification prediction
        class_pred = self.class_model.predict(input_df)[0]
        class_proba = self.class_model.predict_proba(input_df)[0]
        
        # Regression prediction (IC50)
        ic50_pred = self.reg_model.predict(input_df)[0]
        
        # SHAP values
        shap_values = None
        if self.explainer is not None:
            shap_values = self.explainer.shap_values(input_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap_values = shap_values[0]
        
        return class_pred, class_proba, ic50_pred, shap_values


def main():
    st.markdown('<h1 class="main-header">🧪 Project Heisenberg</h1>', unsafe_allow_html=True)
    st.markdown('<div class="heisenberg-tag">I am the one who predicts.</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    predictor = DrugResponsePredictor()
    if not predictor.loaded:
        st.error("⚠️ Models not loaded. Run training pipeline first.")
        return
    
    st.sidebar.title("🔬 Navigation")
    page = st.sidebar.radio("Go to", 
                            ["🏠 Home", "🔮 Prediction", "📊 Model Performance", "📈 Analytics"])
    
    if page == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**🎯 Accuracy**")
            st.metric("Best Classifier", f"{predictor.model_info['best_classification_accuracy']*100:.2f}%", "Classification")
        with col2:
            st.info("**🧬 Features**")
            st.metric("Genomic + Clinical", len(predictor.feature_names), "Input Features")
        with col3:
            st.info("**📉 IC50 R²**")
            st.metric("Best Regressor", f"{predictor.model_info['best_regression_r2']:.4f}", "Regression")
        
        st.markdown("---")
        st.header("📋 About Project Heisenberg")
        st.write("""
        **"Chemistry is the study of matter, but I prefer to see it as the study of change."**  
        — Walter White

        This system predicts drug response using:
        - **Genomic mutations** (TP53, KRAS, EGFR, etc.)
        - **Clinical parameters** (Age, BMI, Stage)
        - **Drug molecular fingerprints** (Morgan fingerprints)
        - **Dual Output**: Responder classification + IC50 regression

        Built with precision. Engineered for accuracy.  
        **Say my name.**
        """)
    
    elif page == "🔮 Prediction":
        st.header("🔮 Predict Drug Response")
        st.write("Enter patient details to determine treatment efficacy")
        
        with st.form("prediction_form"):
            st.subheader("👤 Clinical Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.slider("Age", 20, 85, 50)
                bmi = st.slider("BMI", 15.0, 45.0, 25.0, 0.1)
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female"])
                stage = st.selectbox("Cancer Stage", ["I", "II", "III", "IV"])
            with col3:
                tissue = st.selectbox("Tissue Type", ["Lung", "Breast", "Colon", "Skin", "Blood"])
                prior_treatment = st.checkbox("Prior Treatment")
            
            st.markdown("---")
            st.subheader("🧬 Genomic Mutations")
            genes = ['TP53', 'KRAS', 'EGFR', 'BRAF', 'PIK3CA', 
                    'PTEN', 'APC', 'RB1', 'BRCA1', 'BRCA2', 'MYC', 'NRAS', 'ALK', 'RET', 'MET']
            mutations = {}
            cols = st.columns(5)
            for i, gene in enumerate(genes):
                with cols[i % 5]:
                    mutations[f'MUT_{gene}'] = 1 if st.checkbox(f"{gene}") else 0
            
            st.markdown("---")
            st.subheader("💊 Drug Selection")
            drug_name = st.selectbox("Select Drug", 
                                    ["Erlotinib", "Gefitinib", "Vemurafenib", "Trametinib",
                                     "Pembrolizumab", "Cisplatin", "Docetaxel", "5-FU"])
            
            submit = st.form_submit_button("🔬 Predict Response", use_container_width=True)
        
        if submit:
            input_data = {
                'AGE': age,
                'BMI': bmi,
                'PRIOR_TREATMENT': 1 if prior_treatment else 0,
                'GENDER': gender,
                'STAGE': stage,
                'TISSUE': tissue,
                'DRUG_NAME': drug_name,
                **mutations
            }
            
            with st.spinner("🔄 Analyzing genomic signature..."):
                class_pred, class_proba, ic50_pred, shap_vals = predictor.predict_with_shap(input_data)
            
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if class_pred == 1:
                    st.markdown(
                        '<div class="prediction-box responder">✅ RESPONDER<br/>Treatment likely effective</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="prediction-box non-responder">⚠️ NON-RESPONDER<br/>Consider alternative therapy</div>',
                        unsafe_allow_html=True
                    )
            with col2:
                st.metric("Confidence", f"{max(class_proba)*100:.1f}%", "Probability")
            with col3:
                st.markdown(
                    f'<div class="prediction-box ic50-box">🔬 IC50<br/>{ic50_pred:.2f} μM</div>',
                    unsafe_allow_html=True
                )
            
            st.subheader("📈 Response Probability")
            fig = go.Figure(data=[
                go.Bar(x=['Non-Responder', 'Responder'], y=class_proba,
                      marker_color=['#e74c3c', '#2ecc71'])
            ])
            fig.update_layout(title="Class Probabilities", yaxis_title="Probability", height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            if shap_vals is not None:
                st.markdown("---")
                st.subheader("🔍 Why This Prediction? (SHAP Explainability)")
                st.write("Features pushing the prediction toward **Responder (🔴)** or **Non-Responder (🔵)**")
                
                shap_df = pd.DataFrame({
                    'feature': predictor.feature_names,
                    'shap_value': shap_vals
                }).sort_values('shap_value', key=abs, ascending=False)
                
                top_shap = shap_df.head(10)
                fig_shap = px.bar(
                    top_shap,
                    x='shap_value',
                    y='feature',
                    orientation='h',
                    color='shap_value',
                    color_continuous_scale='RdBu',
                    title="Top 10 Features Influencing Prediction"
                )
                fig_shap.update_layout(height=500)
                st.plotly_chart(fig_shap, use_container_width=True)
    
    elif page == "📊 Model Performance":
        st.header("📊 Model Performance – Precision Engineered")
        
        # Classification results
        try:
            class_df = pd.read_csv(predictor.models_dir / 'classification_comparison.csv')
            st.subheader("Classification Models")
            st.dataframe(class_df.style.background_gradient(
                subset=['Accuracy', 'ROC-AUC'], cmap='RdYlGn'
            ).format({
                'Accuracy': '{:.4f}',
                'ROC-AUC': '{:.4f}',
                'Time(s)': '{:.2f}'
            }), use_container_width=True)
        except:
            st.warning("Classification results not found")
        
        # Regression results
        try:
            reg_df = pd.read_csv(predictor.models_dir / 'regression_comparison.csv')
            st.subheader("Regression Models (IC50 Prediction)")
            st.dataframe(reg_df.style.background_gradient(
                subset=['R2'], cmap='RdYlGn'
            ).format({
                'R2': '{:.4f}',
                'MSE': '{:.4f}',
                'Time(s)': '{:.2f}'
            }), use_container_width=True)
        except:
            st.warning("Regression results not found")
    
    elif page == "📈 Analytics":
        st.header("📈 System Analytics")
        try:
            X_test = pd.read_csv(predictor.data_dir / 'X_test.csv')
            y_test = pd.read_csv(predictor.data_dir / 'y_class_test.csv').values.ravel()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Samples", len(X_test))
            with col2: st.metric("Features", X_test.shape[1])
            with col3: st.metric("Responders", int(y_test.sum()))
            with col4: st.metric("Non-Responders", int((y_test == 0).sum()))
            
            st.subheader("🎯 Response Distribution")
            fig = go.Figure(data=[go.Pie(
                labels=['Non-Responder', 'Responder'],
                values=[(y_test == 0).sum(), y_test.sum()],
                marker_colors=['#e74c3c', '#2ecc71']
            )])
            fig.update_layout(title="Patient Response Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")
    
    st.markdown("""
    <div class="footer-quote">
        "I did it for me. I liked it. I was good at it. And I was really... I was alive."<br>
        — Walter White, Breaking Bad
    </div>
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9em; margin-top: 1rem;">
        Project Heisenberg • Drug Response Prediction System • Developed by Heisenberg
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
