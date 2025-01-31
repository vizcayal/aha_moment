# Enhanced Data Preprocessing Script
# This script includes additional preprocessing techniques or improvements.

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset
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
def main():
    # Load data
    data = load_data('data/extended_dataset.csv')
    
    # Normalize data
    normalized_data = normalize_data(data)
    
    # Standardize data
    standardized_data = standardize_data(data)
    
    # Save processed data
    pd.DataFrame(normalized_data).to_csv('data/normalized_data.csv', index=False)
    pd.DataFrame(standardized_data).to_csv('data/standardized_data.csv', index=False)

if __name__ == '__main__':
    main()
