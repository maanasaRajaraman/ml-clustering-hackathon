import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data = pd.read_csv("data.csv")
selected_features = data[['Spike shape', 'Spike sterility ', 'Spike density ', 'Seed colour', 'Seed shape ', 
                          'Days to 50% flowering', 'Node pigmentation', 'Leaf sheath length', 'Leaf blade length',
                          'Leaf blade width', 'Spike length', 'Spike girth', 'Plant height', '1000 Seed weight', 
                          'YIELD']]

num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(selected_features)
cluster_labels = kmeans.labels_
data['Cluster'] = cluster_labels

print("Cluster Centers:")
print(kmeans.cluster_centers_)

for cluster in range(num_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Spike shape'], cluster_data['Spike sterility '], label=f'Cluster {cluster + 1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=10, c='black', label='Centroids')
plt.xlabel('Spike shape')
plt.ylabel('Spike sterility')
plt.legend()
plt.show()
