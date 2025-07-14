import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import pickle


#----------------------------------------------- Random Split ----------------------------------------
# Split train and test sets while maintaining label proportions
train_set, test_set = train_test_split(genePT_w_df, test_size=0.1, stratify=genePT_w_df.iloc[:, -1], random_state=42)
#------------------------------------------------------------------------------------------------

# Create dataset and data loader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = torch.tensor(self.data.iloc[index, :-1].values, dtype=torch.float32)
        label = torch.tensor(self.data.iloc[index, -1], dtype=torch.long)
        return features, label, index

# Initialize KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model
X_train = train_set.iloc[:, :-1].values
y_train = train_set.iloc[:, -1].values
knn.fit(X_train, y_train)

# Make predictions on test set
X_test = test_set.iloc[:, :-1].values
y_test = test_set.iloc[:, -1].values
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)[:, 1]  # Get predicted probabilities for positive class

# Calculate metrics
accuracy_test = accuracy_score(y_test, y_pred)
precision_test = precision_score(y_test, y_pred)
recall_test = recall_score(y_test, y_pred)
f1_test = f1_score(y_test, y_pred)
roc_auc_test = roc_auc_score(y_test, y_prob)

# Print test set metrics
print("Test Accuracy:", accuracy_test)
print("Test Precision:", precision_test)
print("Test Recall:", recall_test)
print("Test F1-score:", f1_test)
print("Test AUC:", roc_auc_test)