# Enhanced Data Preprocessing Script
# This script includes additional preprocessing techniques or improvements.

import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    logging.info("Loading data...")
def load_data(file_path):
    return pd.read_csv(file_path)

# Normalize data
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

# Standardize data
def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Main function
d    logging.info("Starting data preprocessing...")
ef main():
    # Load data
     logging.info("Data loaded successfully.")
   data = load_data('data/extended_dataset.csv')
    
    # Normalize data
     logging.info("Data normalized.")
   normalized_data = normalize_data(data)
    
    # Standardize data
     logging.info("Data standardized.")
   standardized_data = standardize_data(data)
    
    # Save processed data
    pd.DataFrame(normalized_data).to_csv('data/normalized_data.csv', index=False)
    pd.DataFrame(standardized_data).to_csv('data/standardized_data.csv', index=False)
    logging.info("Data preprocessing completed and saved.")

if __name__ == '__main__':
    main()
