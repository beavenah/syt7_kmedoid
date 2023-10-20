import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
import sys

# Load pairwise RMSD data from the CSV file
pairwise_rmsd = pd.read_csv('./data/rmsd_metric.csv', header=None).values

# elbow method to determine the optimal number of clusters
inertia = []
k_values = range(2, 10)  # You can adjust this based on your dataset
for k in k_values:
    print(k)
    kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=0, method='pam')
    kmedoids.fit(pairwise_rmsd)
    inertia.append(kmedoids.inertia_)

# k-2 because start with k=2. could adjust
for k in k_values:
    print(k, inertia[k-2])

sys.exit()

# Plot the within-cluster sum of squares (WCSS) for different cluster numbers
plt.plot(range(1, max_clusters_to_try + 1), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()
