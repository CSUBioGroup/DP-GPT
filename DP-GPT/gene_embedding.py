import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import pickle

# -------------------------------------------------gene_embedding--------------------
# 1. Load expression data (CSV file)
df = pd.read_csv('/home/Data/Depression20_normalize_test.csv')

# Assume the last column is labels and other columns are gene expression data
expression_data = df.iloc[:, :-1].values  # Get gene expression data for all samples (excluding the last label column)
labels = df.iloc[:, -1].values  # Get labels (last column)

# Gene names correspond to DataFrame column names (assuming column names are gene names)
gene_names = df.columns[:-1].tolist()

# 2. Load GenePT-3.5 gene embedding data
with open("/Data/gene_embedding_ada_test.pickle", "rb") as fp:
    GPT_3_5_gene_embeddings = pickle.load(fp)

# 3. Initialize a matrix to store gene embeddings
EMBED_DIM = 1536  # GPT-3.5 embedding dimension
lookup_embed = []

# Store all matched gene names
matched_gene_names = []

count_missing = 0  # Counter for genes without embeddings

# 4. Find embedding vector for each gene
for i, gene in enumerate(gene_names):
    if gene in GPT_3_5_gene_embeddings:
        embedding = np.array(GPT_3_5_gene_embeddings[gene])  # Ensure it's a numpy array
        lookup_embed.append(embedding.flatten())  # Use flatten()
        matched_gene_names.append(gene)
    else:
        count_missing += 1

# Print number of unmatched genes
print(f"Unable to match {count_missing} out of {len(gene_names)} genes in the GenePT-w embedding")

# 5. Remove expression data for unmatched genes
# Create new expression_data containing only matched genes
matched_gene_indices = [i for i, gene in enumerate(gene_names) if gene in matched_gene_names]
expression_data_filtered = expression_data[:, matched_gene_indices]

# 6. Use mask to ignore missing values (zeros)
# Create mask to mark which gene expression values are valid (non-zero)
train_mask = expression_data_filtered != 0  # Mark non-zero gene expression values

# 7. Calculate genePT-w embedding using mask
# Compute weighted average for each sample's gene expression data, mask ignores zeros
genePT_w_emebed = np.dot(train_mask * expression_data_filtered, lookup_embed) / np.sum(train_mask, axis=1)[:, None]

# 8. Combine genePT_w_emebed and labels
# Convert genePT_w_emebed to DataFrame
genePT_w_df = pd.DataFrame(genePT_w_emebed)
genePT_w_df['label'] = labels