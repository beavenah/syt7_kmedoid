import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids

# process each line and extract required data
def process_line(line):
    tokens = line.strip().split()
    protein_com_coords = [float(x) for x in tokens[2:5]]
    cbl_com_coords = [float(x) for x in tokens[7:10]]
    apx_com_coords = [float(x) for x in tokens[12:15]]
    chl1_coords = [float(x) for x in tokens[17:20]]
    chl2_coords = [float(x) for x in tokens[20:23]]
    segment_id, replica_id, dcd_id, frame_id, chl1_id, chl2_id = tokens[24], int(tokens[25]), int(tokens[26]), int(tokens[27]), int(tokens[28]), int(tokens[29])
    
    return protein_com_coords + cbl_com_coords + apx_com_coords + chl1_coords + chl2_coords + [segment_id, replica_id, dcd_id, frame_id, chl1_id, chl2_id]

# process a file and return processed data list
def process_file(filename):
    with open(filename, 'r') as file:
        return [process_line(line) for line_number, line in enumerate(file, start=1) if not line.startswith('<')]

# read coord data and other info
data = process_file('./data/coordinates.txt')
column_labels = ["pcom_x", "pcom_y", "pcom_z",
                 "cbl_x", "cbl_y", "cbl_z",
                 "apx_x", "apx_y", "apx_z",
                 "chl1_x", "chl1_y", "chl1_z",
                 "chl2_x", "chl2_y", "chl2_z",
                 "segment_id", "replica_id", "dcd_id", "frame_id", "chl1_id", "chl2_id"]

# convert to pandas dataframe
data_df = pd.DataFrame(data, columns=column_labels)

# read rmsd matrix
rmsd_matrix = pd.read_csv('data/rmsd_metric.csv', header=None).values

# do the k-medoid clustering
num_clusters = 6
kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', init='k-medoids++', method='pam').fit(rmsd_matrix)
# labels = cluster id
labels = kmedoids.labels_
# indices = medoid cluster centers
medoid_indices = kmedoids.medoid_indices_

# write the cluster information into the dataframe
data_df['cluster_id'] = labels
data_df.to_csv("./data/clustered_data.tsv", sep='\t', index=False)

# count occurrences for each cluster
occurrences = data_df['cluster_id'].value_counts()

# sort occurrences in descending order to get the order of prevalence
sorted_clusters = occurrences.sort_values(ascending=False).index

# print occurrences in the order of prevalence
for cluster_id in sorted_clusters:
    print(f"Cluster {cluster_id} occurrences: {occurrences[cluster_id]}")

# print medoid coordinates in the order of prevalence
print("\nMedoid Coordinates in order of prevalence:")
for cluster_id in sorted_clusters:
    # find the index for the medoid of this cluster
    idx = medoid_indices[cluster_id]
    print(f"Cluster {cluster_id}:\n", data_df.iloc[[idx]], "\n\n")

medoid_df = pd.concat([data_df.iloc[[idx]] for cluster_id in sorted_clusters])
medoid_df.to_csv("./data/medoids.tsv", sep='\t', index=False)

# print the first few rows of each cluster in the order of prevalence
for cluster_id in sorted_clusters:
    print(f"Cluster {cluster_id}:\n", data_df[data_df['cluster_id'] == cluster_id].head(), "\n\n")
