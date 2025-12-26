# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:51:18 2025

@author: youssef.salman
"""
import numpy as np
import matplotlib.pyplot as plt

# Surface and its gradient
def z_function(x, y):
    return np.sin(5 * x) * np.cos(5 * y) / 5

def calculate_gradient(x, y):
    dz_dx = np.cos(5 * x) * np.cos(5 * y)       # ∂z/∂x
    dz_dy = -np.sin(5 * x) * np.sin(5 * y)     # ∂z/∂y
    return dz_dx, dz_dy

# Grid for plotting the surface
x = np.arange(-1, 1, 0.05)
y = np.arange(-1, 1, 0.05)
X, Y = np.meshgrid(x, y)
Z = z_function(X, Y)

# Gradient descent setup
current_pos = np.array([0.7, 0.4, z_function(0.7, 0.4)], dtype=float)
learning_rate = 0.01
iters = 1000

# Plot / animation
plt.ion()  # NEW: turn on interactive mode so we can animate frame-by-frame

fig = plt.figure()  # NEW: create a Figure explicitly (needed to add 3D axes cleanly)
ax = fig.add_subplot(projection="3d")  # NEW: add a 3D subplot (Axes3D) to the figure

# NEW: fix axes limits so the scene stays stable during animation
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(Z.min(), Z.max())

# NEW: label axes for clarity
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.view_init(elev=30, azim=-60)  # NEW: choose a nice 3D camera angle

for _ in range(iters):
    gx, gy = calculate_gradient(current_pos[0], current_pos[1])  # gradient at current (x,y)

    # Gradient descent step in the x–y plane
    new_x = current_pos[0] - learning_rate * gx
    new_y = current_pos[1] - learning_rate * gy

    new_x = np.clip(new_x, -1, 1)  # NEW: keep point inside plotting domain on x
    new_y = np.clip(new_y, -1, 1)  # NEW: keep point inside plotting domain on y

    new_z = z_function(new_x, new_y)  # recompute z on the surface
    current_pos[:] = (new_x, new_y, new_z)

    # Draw current frame
    ax.plot_surface(
        X, Y, Z, cmap="viridis", linewidth=0, antialiased=True,
        zorder=0, alpha=0.95  # NEW: alpha makes the point visible; zorder draws surface behind the point
    )
    ax.scatter(
        current_pos[0], current_pos[1], current_pos[2],
        color="magenta", s=40, zorder=1  # NEW: draw the moving point above the surface
    )

    plt.pause(0.01)  # NEW: brief pause to render the frame (creates the animation)
    ax.cla()         # NEW: clear only the axes (not the whole figure) for the next frame

# Final static frame (so the last state remains visible)
ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
ax.scatter(current_pos[0], current_pos[1], current_pos[2], color="magenta", s=40)
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(Z.min(), Z.max())  # reapply fixed limits
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.view_init(elev=30, azim=-60)
plt.ioff()  # NEW: turn interactive mode off (back to normal plotting)
plt.show()

