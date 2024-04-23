import pygame
import numpy as np
from sklearn.neighbors import KDTree

# Константы
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

# Параметры DBSCAN
EPS = 30
MIN_SAMPLES = 3


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.flag = None
        self.cluster = None


def draw_points(screen, points):
    for point in points:
        color = BLACK
        if point.flag == 'core':
            color = GREEN
        elif point.flag == 'border':
            color = YELLOW
        elif point.flag == 'noise':
            color = RED
        pygame.draw.circle(screen, color, (point.x, point.y), 5)


def dbscan(points, eps, min_samples):
    X = np.array([(point.x, point.y) for point in points])
    tree = KDTree(X, leaf_size=2)
    clusters = []
    cluster_label = 0

    for point in points:
        if point.cluster is not None:
            continue
        neighbors = tree.query_radius([[point.x, point.y]], r=eps)[0]
        if len(neighbors) < min_samples:
            point.flag = 'noise'
            point.cluster = -1
        else:
            cluster = []
            clusters.append(cluster)
            expand_cluster(points, tree, point, cluster, cluster_label, eps, min_samples)
            cluster_label += 1

    return clusters


def expand_cluster(points, tree, point, cluster, cluster_label, eps, min_samples):
    neighbors = tree.query_radius([[point.x, point.y]], r=eps)[0]
    cluster.append(point)
    point.cluster = cluster_label
    point.flag = 'core'

    i = 0
    while i < len(neighbors):
        neighbor_point = points[neighbors[i]]
        if neighbor_point.flag == 'noise':
            neighbor_point.flag = 'border'
        if neighbor_point.cluster is None:
            cluster.append(neighbor_point)
            neighbor_point.cluster = cluster_label
            neighbor_neighbors = tree.query_radius([[neighbor_point.x, neighbor_point.y]], r=eps)[0]
            if len(neighbor_neighbors) >= min_samples:
                neighbor_point.flag = 'core'
                neighbors = np.append(neighbors, neighbor_neighbors)
            else:
                neighbor_point.flag = 'border'
        i += 1


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DBSCAN Visualization")
    clock = pygame.time.Clock()
    running = True
    points = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                points.append(Point(x, y))
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    clusters = dbscan(points, EPS, MIN_SAMPLES)

        screen.fill(WHITE)
        draw_points(screen, points)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
