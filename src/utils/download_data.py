"""
Data Download Script for GDSC Drug Response Data
File: src/utils/download_data.py
"""

import requests
import pandas as pd
import os
from pathlib import Path

class GDSCDataDownloader:
    """Download and prepare GDSC (Genomics of Drug Sensitivity in Cancer) data"""
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # GDSC public data URLs
        self.urls = {
            'drug_response': 'https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/preprocessed/GDSC2_fitted_dose_response_25Feb20.xlsx',
            'cell_line_info': 'https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/preprocessed/Cell_Lines_Details.xlsx',
            'drug_info': 'https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/preprocessed/Screened_Compounds.xlsx'
        }
    
    def download_file(self, url, filename):
        """Download file from URL"""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            print(f"✓ {filename} already exists")
            return filepath
        
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✓ Downloaded {filename}")
            return filepath
        
        except Exception as e:
            print(f"✗ Error downloading {filename}: {str(e)}")
            return None
    
    def download_all(self):
        """Download all GDSC datasets"""
        print("=== Downloading GDSC Datasets ===\n")
        
        downloaded_files = {}
        for name, url in self.urls.items():
            filename = f"{name}.xlsx"
            filepath = self.download_file(url, filename)
            if filepath:
                downloaded_files[name] = filepath
        
        return downloaded_files
    
    def create_sample_data(self):
        """Create sample synthetic data if downloads fail"""
        print("\n=== Creating Sample Synthetic Data ===")
        
        import numpy as np
        
        # Sample drug response data
        n_samples = 1000
        
        data = {
            'COSMIC_ID': np.random.randint(1000000, 9999999, n_samples),
            'DRUG_NAME': np.random.choice(['Drug_A', 'Drug_B', 'Drug_C', 'Drug_D'], n_samples),
            'LN_IC50': np.random.randn(n_samples) * 2 + 3,  # Log IC50 values
            'AUC': np.random.rand(n_samples) * 0.5 + 0.25,
            'RMSE': np.random.rand(n_samples) * 0.3,
            'Z_SCORE': np.random.randn(n_samples)
        }
        
        df_response = pd.DataFrame(data)
        df_response['RESPONSE'] = (df_response['LN_IC50'] < df_response['LN_IC50'].median()).astype(int)
        
        # Sample cell line info
        unique_cells = df_response['COSMIC_ID'].unique()
        cell_data = {
            'COSMIC_ID': unique_cells,
            'CELL_LINE_NAME': [f"Cell_{i}" for i in range(len(unique_cells))],
            'TISSUE': np.random.choice(['Lung', 'Breast', 'Colon', 'Skin', 'Blood'], len(unique_cells)),
            'CANCER_TYPE': np.random.choice(['Carcinoma', 'Sarcoma', 'Leukemia', 'Lymphoma'], len(unique_cells))
        }
        df_cells = pd.DataFrame(cell_data)
        
        # Sample drug info
        unique_drugs = df_response['DRUG_NAME'].unique()
        drug_data = {
            'DRUG_NAME': unique_drugs,
            'DRUG_ID': range(1, len(unique_drugs) + 1),
            'TARGET': np.random.choice(['EGFR', 'BRAF', 'MEK', 'PI3K'], len(unique_drugs)),
            'PATHWAY': np.random.choice(['RTK signaling', 'Cell cycle', 'Apoptosis'], len(unique_drugs))
        }
        df_drugs = pd.DataFrame(drug_data)
        
        # Save sample data
        df_response.to_csv(self.data_dir / 'drug_response_sample.csv', index=False)
        df_cells.to_csv(self.data_dir / 'cell_line_info_sample.csv', index=False)
        df_drugs.to_csv(self.data_dir / 'drug_info_sample.csv', index=False)
        
        print("✓ Created sample datasets in data/raw/")
        print(f"  - drug_response_sample.csv ({len(df_response)} records)")
        print(f"  - cell_line_info_sample.csv ({len(df_cells)} cell lines)")
        print(f"  - drug_info_sample.csv ({len(df_drugs)} drugs)")
        
        return {
            'response': self.data_dir / 'drug_response_sample.csv',
            'cells': self.data_dir / 'cell_line_info_sample.csv',
            'drugs': self.data_dir / 'drug_info_sample.csv'
        }


def main():
    """Main execution function"""
    downloader = GDSCDataDownloader()
    
    # Try to download real data
    files = downloader.download_all()
    
    # If downloads fail, create sample data
    if len(files) == 0:
        print("\n⚠ Could not download real data. Creating sample synthetic data...")
        sample_files = downloader.create_sample_data()
        print("\n✓ Setup complete! Use sample data for testing.")
    else:
        print("\n✓ All datasets downloaded successfully!")
        print("\nDownloaded files:")
        for name, path in files.items():
            print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
