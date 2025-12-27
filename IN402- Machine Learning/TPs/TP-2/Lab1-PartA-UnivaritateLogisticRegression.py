import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Example dataset: exam score vs admitted (1) / not admitted (0)
X = np.array([30, 40, 50, 60, 70, 80, 90])
y = np.array([0, 0, 0, 1, 1, 1, 1])

plt.scatter(X, y, c=y, cmap="bwr", edgecolors="k")
plt.xlabel("Exam Score")
plt.ylabel("Admitted (1) / Not Admitted (0)")
plt.title("Training Data")
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Test
print(sigmoid(0))  # expected 0.5

def hypothesis(theta0, theta1, x):
    z = theta0 + theta1 * x
    return sigmoid(z)


# Test
print(hypothesis(0, 1, 0))  # expected 0.5

def compute_cost(theta0, theta1, X, y):
    m = len(y)
    predictions = hypothesis(theta0, theta1, X)
    cost = -(1 / m) * np.sum(
        y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
    )
    return cost


# Test
print(compute_cost(0, 0, X, y))  # expected around 0.693

def gradient_descent(X, y, theta0, theta1, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        predictions = hypothesis(theta0, theta1, X)

        error = predictions - y
        theta0 -= alpha * (1 / m) * np.sum(error)
        theta1 -= alpha * (1 / m) * np.sum(error * X)
        cost = compute_cost(theta0, theta1, X, y)
        cost_history.append(cost)

    return theta0, theta1, cost_history

theta0, theta1, cost_history = gradient_descent(
    X, y, theta0=0, theta1=0, alpha=0.01, iterations=1000
)

print("Final theta0:", theta0)
print("Final theta1:", theta1)

# Plot cost convergence
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.title("Cost Function Convergence")
plt.show()

# Plot decision boundary
plt.scatter(X, y, c=y, cmap="bwr", edgecolors="k")
x_vals = np.linspace(min(X), max(X), 100)
plt.plot(x_vals, hypothesis(theta0, theta1, x_vals), color="green")
plt.xlabel("Exam Score")
plt.ylabel("Admitted Probability")
plt.show()