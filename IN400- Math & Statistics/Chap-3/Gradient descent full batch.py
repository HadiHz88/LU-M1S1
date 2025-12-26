# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:40:18 2025

@author: youssef.salman
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the function y = x^2
def y_function(x):
    return x ** 2

# Define its derivative y' = 2x
def y_derivative(x):
    return 2 * x

# Generate x values for plotting
x = np.arange(-100, 100, 0.1)
y = y_function(x)

# Starting point (far from the minimum)
current_pos = (50, y_function(50))

# Learning rate
learning_rate = 0.01

# Make interactive plot
plt.ion()

# Gradient descent iterations
for _ in range(1000):
    # Update rule: x_{new} = x - eta * f'(x)
    new_x = current_pos[0] - learning_rate * y_derivative(current_pos[0])
    new_y = y_function(new_x)
    current_pos = (new_x, new_y)

    # Plot function and current position
    plt.plot(x, y)
    plt.scatter(current_pos[0], current_pos[1], color="red")

    plt.pause(0.001)   # Small delay for animation
    plt.clf()          # Clear figure for next frame

plt.ioff()
plt.show()

