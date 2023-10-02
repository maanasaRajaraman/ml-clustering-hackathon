import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data.csv")
selected_features = data[['Spike shape', 'Spike sterility ', 'Spike density ', 'Seed colour', 'Seed shape ', 
                           'Spike length', 'Spike girth', 'YIELD']]
scaler = StandardScaler()
data_standardized = scaler.fit_transform(selected_features)

data_transpose = data_standardized.T

num_clusters = 20
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data_transpose, num_clusters, m=2, error=0.005, maxiter=1000, seed=0)

cluster_membership = np.argmax(u, axis=0)
data['Cluster'] = cluster_membership

print("Cluster Centers:")
print(cntr)

print("FPC:", fpc)

for cluster in range(num_clusters):
    num_points = np.sum(cluster_membership == cluster)
    print(f"Number of data points in Cluster {cluster + 1}: {num_points}")

feature1 = 'Spike shape'
feature2 = 'Spike sterility '

for cluster in range(num_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data[feature1], cluster_data[feature2], label=f'Cluster {cluster + 1}')

plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()
plt.show()
