"""
Generate Biologically Accurate Synthetic Data with Clear Drug-Gene Rules
File: src/utils/generate_realistic_data.py

Aligned with project objectives: clear responder/non-responder logic,
strong resistance signals, and realistic but learnable patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)


class RealisticDataGenerator:
    """Generate biologically plausible synthetic drug response data"""
    
    def __init__(self, n_samples=1500, data_dir='data/raw'):
        self.n_samples = n_samples
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_data(self):
        print("=== Generating Biologically Accurate Drug Response Data ===\n")
        
        data = {}
        data['COSMIC_ID'] = np.arange(1000000, 1000000 + self.n_samples)
        
        # === CLINICAL FEATURES ===
        print("Generating clinical features...")
        data['AGE'] = np.random.normal(58, 12, self.n_samples).clip(25, 85).astype(int)
        data['BMI'] = np.random.normal(26, 5, self.n_samples).clip(16, 45)
        data['GENDER'] = np.random.choice(['Male', 'Female'], self.n_samples)
        data['STAGE'] = np.random.choice(['I', 'II', 'III', 'IV'], self.n_samples, 
                                        p=[0.15, 0.25, 0.35, 0.25])
        data['TISSUE'] = np.random.choice(['Lung', 'Breast', 'Colon', 'Skin', 'Blood'], 
                                         self.n_samples)
        data['PRIOR_TREATMENT'] = np.random.binomial(1, 0.4, self.n_samples)
        
        # === GENOMIC FEATURES ===
        print("Generating genomic mutations...")
        genes_info = {
            'TP53': 0.50, 'KRAS': 0.25, 'EGFR': 0.20, 'BRAF': 0.15,
            'PIK3CA': 0.30, 'PTEN': 0.20, 'APC': 0.18, 'RB1': 0.12,
            'BRCA1': 0.10, 'BRCA2': 0.08, 'MYC': 0.15, 'NRAS': 0.12,
            'ALK': 0.05, 'RET': 0.04, 'MET': 0.08
        }
        for gene, mut_rate in genes_info.items():
            data[f'MUT_{gene}'] = np.random.binomial(1, mut_rate, self.n_samples)
        
        # === DRUG ASSIGNMENT ===
        print("Assigning drugs...")
        drugs = ['Erlotinib', 'Gefitinib', 'Vemurafenib', 'Trametinib', 
                'Pembrolizumab', 'Cisplatin', 'Docetaxel', '5-FU']
        data['DRUG_NAME'] = np.random.choice(drugs, self.n_samples)
        
        df = pd.DataFrame(data)
        
        # === CREATE RESPONSE WITH STRONG BIOLOGICAL RULES ===
        print("Calculating drug responses with dominant resistance logic...")
        response_score = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            score = 0
            drug = df.loc[i, 'DRUG_NAME']
            
            # 🔴 OVERRIDE: Absolute resistance rules (set score very low)
            resistant = False
            
            # EGFR inhibitors + KRAS/NRAS = resistance
            if drug in ['Erlotinib', 'Gefitinib']:
                if df.loc[i, 'MUT_KRAS'] == 1 or df.loc[i, 'MUT_NRAS'] == 1:
                    score = -10  # Strong resistance
                    resistant = True
            
            # BRAF inhibitor + NRAS = resistance
            if drug == 'Vemurafenib':
                if df.loc[i, 'MUT_NRAS'] == 1:
                    score = -10
                    resistant = True
            
            # If not resistant, apply positive rules
            if not resistant:
                # EGFR inhibitors
                if drug in ['Erlotinib', 'Gefitinib']:
                    if df.loc[i, 'MUT_EGFR'] == 1:
                        score += 4.0  # Strong signal
                    else:
                        score -= 2.0  # No target = likely non-responder
                
                # BRAF inhibitor
                elif drug == 'Vemurafenib':
                    if df.loc[i, 'MUT_BRAF'] == 1:
                        score += 4.0
                    else:
                        score -= 2.0
                
                # MEK inhibitor
                elif drug == 'Trametinib':
                    if df.loc[i, 'MUT_KRAS'] == 1 or df.loc[i, 'MUT_BRAF'] == 1:
                        score += 3.0
                
                # Immunotherapy
                elif drug == 'Pembrolizumab':
                    if df.loc[i, 'TISSUE'] in ['Lung', 'Skin']:
                        score += 2.0
                        if df.loc[i, 'MUT_TP53'] == 1:
                            score += 1.5  # TP53 enhances response in these tissues
                
                # Chemo (baseline)
                elif drug in ['Cisplatin', 'Docetaxel', '5-FU']:
                    score += 1.0
                    if df.loc[i, 'MUT_TP53'] == 1:
                        score += 0.8
                    if df.loc[i, 'PRIOR_TREATMENT'] == 1:
                        score -= 2.0  # Strong resistance from prior treatment
            
            # Clinical modifiers (weaker)
            if df.loc[i, 'STAGE'] == 'IV':
                score -= 1.0
            elif df.loc[i, 'STAGE'] == 'I':
                score += 0.8
            if df.loc[i, 'AGE'] > 70:
                score -= 0.5
            if df.loc[i, 'BMI'] < 18.5 or df.loc[i, 'BMI'] > 35:
                score -= 0.5
            
            response_score[i] = score
        
        # 🔬 Reduced noise for clearer learning
        response_score += np.random.normal(0, 0.3, self.n_samples)  # Was 0.8
        
        # Use median for balanced classes
        threshold = np.median(response_score)
        df['RESPONSE'] = (response_score > threshold).astype(int)
        
        # Generate continuous targets (for future regression)
        df['LN_IC50'] = -response_score + np.random.normal(0, 0.2, self.n_samples)
        df['AUC'] = 1 / (1 + np.exp(response_score)) + np.random.normal(0, 0.05, self.n_samples)
        df['AUC'] = df['AUC'].clip(0, 1)
        
        # Print stats
        print(f"\n✓ Generated {self.n_samples} samples")
        print(f"  Responders: {df['RESPONSE'].sum()} ({df['RESPONSE'].mean()*100:.1f}%)")
        print(f"  Non-responders: {(df['RESPONSE']==0).sum()} ({(df['RESPONSE']==0).mean()*100:.1f}%)")
        print(f"  Unique drugs: {df['DRUG_NAME'].nunique()}")
        print(f"  Features: {len(df.columns)}")
        
        # Save
        output_file = self.data_dir / 'drug_response_realistic.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved: {output_file}")
        
        self._create_supplementary_files(df)
        return df
    
    def _create_supplementary_files(self, df):
        # Cell line info
        cell_data = df[['COSMIC_ID', 'TISSUE', 'STAGE']].drop_duplicates()
        cell_data['CELL_LINE_NAME'] = [f"CELL_{i}" for i in range(len(cell_data))]
        cell_data['CANCER_TYPE'] = cell_data['TISSUE'].map({
            'Lung': 'NSCLC',
            'Breast': 'Adenocarcinoma',
            'Colon': 'Colorectal',
            'Skin': 'Melanoma',
            'Blood': 'Leukemia'
        })
        cell_file = self.data_dir / 'cell_line_info_realistic.csv'
        cell_data.to_csv(cell_file, index=False)
        print(f"✓ Saved: {cell_file}")
        
        # Drug info
        drug_info = {
            'DRUG_NAME': ['Erlotinib', 'Gefitinib', 'Vemurafenib', 'Trametinib',
                         'Pembrolizumab', 'Cisplatin', 'Docetaxel', '5-FU'],
            'TARGET': ['EGFR', 'EGFR', 'BRAF', 'MEK',
                      'PD-1', 'DNA', 'Microtubules', 'Thymidylate'],
            'DRUG_TYPE': ['TKI', 'TKI', 'TKI', 'TKI',
                         'Immunotherapy', 'Chemotherapy', 'Chemotherapy', 'Chemotherapy'],
            'PATHWAY': ['EGFR signaling', 'EGFR signaling', 'MAPK pathway', 'MAPK pathway',
                       'Immune checkpoint', 'DNA damage', 'Cell division', 'DNA synthesis']
        }
        drug_df = pd.DataFrame(drug_info)
        drug_file = self.data_dir / 'drug_info_realistic.csv'
        drug_df.to_csv(drug_file, index=False)
        print(f"✓ Saved: {drug_file}")


def main():
    generator = RealisticDataGenerator(n_samples=1500)
    df = generator.generate_data()
    
    print("\n" + "="*60)
    print("✓ BIOLOGICALLY ACCURATE DATA GENERATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python src/preprocessing/preprocess_data.py")
    print("2. Run: python src/modeling/train_models.py")
    print("3. Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
