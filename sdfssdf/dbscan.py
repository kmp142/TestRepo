import random
import sys

import pygame

import dbscann

pygame.init()
eps = 30
min_pts = 3

WIDTH, HEIGHT = 800, 600

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))

points = []
points_colors = []
default_color = 'black'
colors = ['red', 'green', 'purple', 'yellow', 'magenta', 'purple', 'orange']

running = True
while running:
    screen.fill('WHITE')

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                points.append(event.pos)
                points_colors.append(default_color)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                points = []
                points_colors = []
            elif event.key == pygame.K_r:
                cluster = dbscann.dbscan_algorithm(points, eps=eps, min_samples=min_pts)
                for i in range(len(cluster)):
                    points_colors[i] = colors[cluster[i]]

    for i in range(len(points)):
        pygame.draw.circle(screen, points_colors[i], points[i], 3)

    pygame.display.flip()

pygame.quit()
sys.exit()
