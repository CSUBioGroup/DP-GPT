import os
import pandas as pd
import copy
import numpy as np
import scanpy as sc

# Read CSV file
df = pd.read_csv('home/Data/GEO_MDD/All_MDD.csv')
# Separate label column
labels = df.iloc[:, -1]  # Assuming last column is the label column
data = df.iloc[:, :-1]   # Other columns are data columns
print(data.shape)
# Convert DataFrame to AnnData object
data = data.astype(float)
print(data.dtypes)

adata = sc.AnnData(data)

# Perform preprocessing steps
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata, base=2)

# Get processed data
processed_data = adata.X

# Recombine processed data with labels into DataFrame
processed_df = pd.DataFrame(processed_data, columns=data.columns)  # Convert to DataFrame format
processed_df['label'] = labels  # Add label column

last_column = processed_df.iloc[:, -1]  # Get last column
print(last_column)

# Save processed data to CSV file
output_path = 'home/Data/Depression20_normalize_test.csv'
processed_df.to_csv(output_path, index=False)
