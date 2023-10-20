import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

# Load pairwise RMSD data from the CSV file
pairwise_rmsd = pd.read_csv('./data/rmsd_metric.csv', header=None).values

kmedoids = KMedoids(n_clusters=6, metric='precomputed', random_state=42, method='pam')
labels = kmedoids.fit_predict(pairwise_rmsd)

# Compute silhouette score
score = silhouette_score(pairwise_rmsd, labels, metric='precomputed')
print("Silhouette Score:", score)

