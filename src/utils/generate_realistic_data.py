"""
Generate Realistic Synthetic Data with Patterns
File: src/utils/generate_realistic_data.py

This creates data with actual biological patterns that ML models can learn
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)  # For reproducibility


class RealisticDataGenerator:
    """Generate biologically plausible synthetic drug response data"""
    
    def __init__(self, n_samples=1000, data_dir='data/raw'):
        self.n_samples = n_samples
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_data(self):
        """Generate complete dataset with realistic patterns"""
        print("=== Generating Realistic Drug Response Data ===\n")
        
        # Initialize data storage
        data = {}
        
        # Generate patient IDs
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
        
        # === GENOMIC FEATURES (with biological relevance) ===
        print("Generating genomic mutations...")
        
        # Key cancer genes with realistic mutation rates
        genes_info = {
            'TP53': 0.50,      # Most commonly mutated
            'KRAS': 0.25,      # Common in lung/colon
            'EGFR': 0.20,      # Targetable mutation
            'BRAF': 0.15,      # MAPK pathway
            'PIK3CA': 0.30,    # PI3K pathway
            'PTEN': 0.20,      # Tumor suppressor
            'APC': 0.18,       # Colon cancer
            'RB1': 0.12,       # Cell cycle
            'BRCA1': 0.10,     # DNA repair
            'BRCA2': 0.08,     # DNA repair
            'MYC': 0.15,       # Oncogene
            'NRAS': 0.12,      # RAS pathway
            'ALK': 0.05,       # Fusion gene
            'RET': 0.04,       # Targetable
            'MET': 0.08        # Growth factor receptor
        }
        
        for gene, mut_rate in genes_info.items():
            data[f'MUT_{gene}'] = np.random.binomial(1, mut_rate, self.n_samples)
        
        # === DRUG SELECTION ===
        print("Assigning drugs...")
        drugs = ['Erlotinib', 'Gefitinib', 'Vemurafenib', 'Trametinib', 
                'Pembrolizumab', 'Cisplatin', 'Docetaxel', '5-FU']
        data['DRUG_NAME'] = np.random.choice(drugs, self.n_samples)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # === CREATE RESPONSE BASED ON BIOLOGICAL RULES ===
        print("Calculating drug responses based on biology...")
        response_score = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            score = 0
            
            # Rule 1: EGFR inhibitors work with EGFR mutations
            if df.loc[i, 'DRUG_NAME'] in ['Erlotinib', 'Gefitinib']:
                if df.loc[i, 'MUT_EGFR'] == 1:
                    score += 3.0  # Strong positive effect
                if df.loc[i, 'MUT_KRAS'] == 1:
                    score -= 2.0  # KRAS mutation = resistance
            
            # Rule 2: BRAF inhibitors work with BRAF mutations
            if df.loc[i, 'DRUG_NAME'] == 'Vemurafenib':
                if df.loc[i, 'MUT_BRAF'] == 1:
                    score += 3.5
                if df.loc[i, 'MUT_NRAS'] == 1:
                    score -= 1.5  # NRAS = resistance
            
            # Rule 3: MEK inhibitors
            if df.loc[i, 'DRUG_NAME'] == 'Trametinib':
                if df.loc[i, 'MUT_KRAS'] == 1 or df.loc[i, 'MUT_BRAF'] == 1:
                    score += 2.5
            
            # Rule 4: Immunotherapy works better with certain profiles
            if df.loc[i, 'DRUG_NAME'] == 'Pembrolizumab':
                if df.loc[i, 'MUT_TP53'] == 1:
                    score += 1.5
                if df.loc[i, 'TISSUE'] in ['Lung', 'Skin']:
                    score += 1.0
            
            # Rule 5: Traditional chemo - less specific but baseline effective
            if df.loc[i, 'DRUG_NAME'] in ['Cisplatin', 'Docetaxel', '5-FU']:
                score += 1.0
                if df.loc[i, 'MUT_TP53'] == 1:
                    score += 0.8
                if df.loc[i, 'PRIOR_TREATMENT'] == 1:
                    score -= 1.5  # Resistance from prior treatment
            
            # Rule 6: Clinical factors
            if df.loc[i, 'STAGE'] == 'IV':
                score -= 1.0  # Advanced stage = worse response
            elif df.loc[i, 'STAGE'] == 'I':
                score += 0.8  # Early stage = better response
            
            if df.loc[i, 'AGE'] > 70:
                score -= 0.5  # Older age = slightly worse
            
            if df.loc[i, 'BMI'] < 18.5 or df.loc[i, 'BMI'] > 35:
                score -= 0.5  # Extreme BMI = worse response
            
            # Rule 7: TP53 mutation generally indicates aggressive cancer
            if df.loc[i, 'MUT_TP53'] == 1:
                score -= 0.3
            
            # Rule 8: PI3K pathway alterations
            if df.loc[i, 'MUT_PIK3CA'] == 1 or df.loc[i, 'MUT_PTEN'] == 1:
                score -= 0.4  # Often leads to resistance
            
            response_score[i] = score
        
        # Add some random noise (biology is not perfectly predictable)
        response_score += np.random.normal(0, 0.8, self.n_samples)
        
        # Convert to binary response (1 = responder, 0 = non-responder)
        # Use median as threshold for balanced classes
        threshold = np.median(response_score)
        df['RESPONSE'] = (response_score > threshold).astype(int)
        
        # Generate IC50 values based on response score
        # Lower IC50 = more sensitive = responder
        df['LN_IC50'] = -response_score + np.random.normal(0, 0.5, self.n_samples)
        df['AUC'] = 1 / (1 + np.exp(response_score)) + np.random.normal(0, 0.1, self.n_samples)
        df['AUC'] = df['AUC'].clip(0, 1)
        
        # Print statistics
        print(f"\n✓ Generated {self.n_samples} samples")
        print(f"  Responders: {df['RESPONSE'].sum()} ({df['RESPONSE'].mean()*100:.1f}%)")
        print(f"  Non-responders: {(df['RESPONSE']==0).sum()} ({(df['RESPONSE']==0).mean()*100:.1f}%)")
        print(f"  Unique drugs: {df['DRUG_NAME'].nunique()}")
        print(f"  Features: {len(df.columns)}")
        
        # Save data
        output_file = self.data_dir / 'drug_response_realistic.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved: {output_file}")
        
        # Create cell line and drug info files
        self._create_supplementary_files(df)
        
        return df
    
    def _create_supplementary_files(self, df):
        """Create cell line and drug information files"""
        
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
    """Main execution"""
    generator = RealisticDataGenerator(n_samples=1500)
    df = generator.generate_data()
    
    print("\n" + "="*60)
    print("✓ REALISTIC DATA GENERATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python src/preprocessing/preprocess_data.py")
    print("2. Run: python src/modeling/train_models.py")
    print("3. Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
