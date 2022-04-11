import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# 1. Найти оптимальное количество кластеров при помощи готовых библиотек (sklearn).

iris = load_iris()
X = iris.data

silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score for different numbers of clusters')
plt.show()

optimal_num_clusters = np.argmax(silhouette_scores) + 2
print("Оптимальное количество кластеров:", optimal_num_clusters)
