#Gaussian Mixture Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

data = pd.read_csv("data.csv")

selected_features = data[['Spike shape', 'Spike sterility ', 'Spike density ', 'Seed colour', 'Seed shape ',
                          'Days to 50% flowering', 'Node pigmentation', 'Leaf sheath length', 'Leaf blade length',
                          'Leaf blade width', 'Spike length', 'Spike girth', 'Plant height', '1000 Seed weight',
                          'YIELD']]

scaler = StandardScaler()
data_standardized = scaler.fit_transform(selected_features)

num_clusters = 15

gmm = GaussianMixture(n_components=num_clusters, random_state=0)
cluster_labels = gmm.fit_predict(data_standardized)
data['Cluster'] = cluster_labels

feature1 = 'Spike shape'
feature2 = 'Spike sterility '

for cluster in range(num_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data[feature1], cluster_data[feature2], label=f'Cluster {cluster + 1}')

plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()
plt.title(f'Scatter Plot of {feature1} vs {feature2} (GMM)')
plt.show()
