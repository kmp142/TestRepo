import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 2. Написать самостоятельно алгоритм для
# оптимального количества кластеров (п.1) k-means без использования
# библиотек (но можно пользоваться библиотеками, не связанными с самим
# алгоритмом - отрисовки, подсчетов и т.д.).
# Рисунки выводятся на каждый шаг – сдвиг центроидов, смена точек
# своих кластеров. Сколько шагов, столько рисунков (можно в виде gif).
# Точки из разных кластеров разными цветами.


iris = load_iris()
X = iris.data

def plot_clusters(X, centroids, labels, title):
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b']
    for i in range(len(centroids)):
        plt.scatter(X[labels == i][:, 0], X[labels == i][:, 1], color=colors[i], label=f'Cluster {i}')
        plt.scatter(centroids[i][0], centroids[i][1], color=colors[i], marker='x', s=200)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def kmeans_custom(X, n_clusters, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    for _ in range(max_iters):
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

        plot_clusters(X, centroids, labels, f'Step {_ + 1}')

    return centroids, labels

optimal_num_clusters = 3
centroids, labels = kmeans_custom(X[:, :2], optimal_num_clusters)

plot_clusters(X[:, :2], centroids, labels, 'Final Clustering Result')
