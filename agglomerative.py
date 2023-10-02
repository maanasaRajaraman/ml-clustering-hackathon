import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import linkage, dendrogram

data = pd.read_csv("data.csv")
selected_features = data[['Spike shape', 'Spike sterility ', 'Spike density ', 'Seed colour', 'Seed shape ', 
                           'Spike length', 'Spike girth', 'YIELD']]
scaler = StandardScaler()
data_standardized = scaler.fit_transform(selected_features)
num_clusters = 20
model = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
cluster_labels = model.fit_predict(data_standardized)
data['Cluster'] = cluster_labels

#visualized using a dendrogram

linked = linkage(data_standardized, method='ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

#visualized using scatter plot
feature1 = 'Spike shape'
feature2 = 'Spike sterility '

for cluster in range(num_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data[feature1], cluster_data[feature2], label=f'Cluster {cluster + 1}')

plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()
plt.show()
