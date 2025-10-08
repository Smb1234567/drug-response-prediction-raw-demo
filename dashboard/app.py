"""
Drug Response Prediction Dashboard – Project Heisenberg
File: dashboard/app.py
Run with: streamlit run dashboard/app.py

Developed by: Heisenberg
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Project Heisenberg: Drug Response Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Heisenberg theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .heisenberg-tag {
        text-align: center;
        color: #e74c3c;
        font-weight: bold;
        margin-bottom: 1.5rem;
        font-size: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .responder {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .non-responder {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .footer-quote {
        font-style: italic;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 2px solid #bdc3c7;
    }
</style>
""", unsafe_allow_html=True)


class DrugResponsePredictor:
    """Drug response prediction system – Project Heisenberg"""
    
    def __init__(self, models_dir='models', data_dir='data/processed'):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.load_artifacts()
    
    def load_artifacts(self):
        try:
            with open(self.models_dir / 'best_model_info.json', 'r') as f:
                self.model_info = json.load(f)
            
            best_model_name = self.model_info['best_model']
            model_filename = f"{best_model_name.replace(' ', '_').lower()}_model.pkl"
            self.model = joblib.load(self.models_dir / model_filename)
            
            X_test = pd.read_csv(self.data_dir / 'X_test.csv')
            self.feature_names = X_test.columns.tolist()
            
            self.comparison_df = pd.read_csv(self.models_dir / 'model_comparison.csv')
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
        
        return encoded_input
    
    def predict(self, input_data):
        encoded_input = self._create_encoded_features(input_data)
        input_df = pd.DataFrame([encoded_input])
        input_df = input_df[self.feature_names]
        
        prediction = self.model.predict(input_df)[0]
        probability = self.model.predict_proba(input_df)[0]
        return prediction, probability


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
            st.metric("Best Model", f"{predictor.model_info['best_accuracy']*100:.2f}%", predictor.model_info['best_model'])
        with col2:
            st.info("**🧬 Features**")
            st.metric("Genomic + Clinical", len(predictor.feature_names), "Input Features")
        with col3:
            st.info("**🏆 Algorithm**")
            st.metric("Top Performer", predictor.model_info['best_model'], "Selected")
        
        st.markdown("---")
        st.header("📋 About Project Heisenberg")
        st.write("""
        **"Chemistry is the study of matter, but I prefer to see it as the study of change."**  
        — Walter White

        This system predicts drug response using:
        - **Genomic mutations** (TP53, KRAS, EGFR, etc.)
        - **Clinical parameters** (Age, BMI, Stage)
        - **Drug selection** (Targeted therapies & chemotherapies)

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
                prediction, probability = predictor.predict(input_data)
            
            st.markdown("---")
            st.header("📊 Prediction Result")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if prediction == 1:
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
                st.metric("Confidence", f"{max(probability)*100:.1f}%", "Probability")
            
            st.subheader("📈 Response Probability")
            fig = go.Figure(data=[
                go.Bar(x=['Non-Responder', 'Responder'], y=probability,
                      marker_color=['#e74c3c', '#2ecc71'])
            ])
            fig.update_layout(title="Class Probabilities", yaxis_title="Probability", height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📊 Model Performance":
        st.header("📊 Model Performance – Precision Engineered")
        styled_df = predictor.comparison_df.style.background_gradient(
            subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            cmap='RdYlGn'
        ).format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}',
            'ROC-AUC': '{:.4f}',
            'Time(s)': '{:.2f}'
        })
        st.dataframe(styled_df, use_container_width=True)
        
        best = predictor.comparison_df.iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("🥇 Best Model", best['Model'])
        with col2: st.metric("Accuracy", f"{best['Accuracy']:.4f}")
        with col3: st.metric("ROC-AUC", f"{best['ROC-AUC']:.4f}")
    
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
    
    # === HEISENBERG FOOTER ===
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