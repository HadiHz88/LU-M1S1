import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: DATA PREPARATION
# -----------------------------

# Simple dataset: House Size (X) vs. Price (y)
X = np.array([800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000])
y = np.array(
    [150000, 180000, 200000, 230000, 260000, 280000, 300000, 330000, 360000, 380000]
)

# Visualize the data
plt.scatter(X, y, color="blue")
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price ($)")
plt.title("House Size vs Price")
plt.show()


# -----------------------------
# STEP 2: HYPOTHESIS FUNCTION
# -----------------------------
# h(x) = theta0 + theta1 * x
def hypothesis(X, theta0, theta1):
    return theta0 + theta1 * X


# -----------------------------
# STEP 3: COST FUNCTION
# -----------------------------
# Mean Squared Error (MSE)
def computeCost(X, y, theta0, theta1):
    m = len(y)
    predictions = hypothesis(X, theta0, theta1)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


# -----------------------------
# STEP 4A: BATCH GRADIENT DESCENT
# -----------------------------
# Updates theta0, theta1 once per full dataset
def batch_gradient_descent(X, y, theta0, theta1, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        predictions = hypothesis(X, theta0, theta1)
        error = predictions - y

        # Compute gradient using all data points
        theta0 -= alpha * (1 / m) * np.sum(error)
        theta1 -= alpha * (1 / m) * np.sum(error * X)

        # Track cost after each iteration
        cost_history.append(computeCost(X, y, theta0, theta1))

    return theta0, theta1, cost_history


# -----------------------------
# STEP 4B: STOCHASTIC GRADIENT DESCENT
# -----------------------------
# Updates theta0, theta1 after each single data point
def stochastic_gradient_descent(X, y, theta0, theta1, alpha, epochs):
    m = len(y)
    cost_history = []

    for epoch in range(epochs):
        # Shuffle data to make learning less biased
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(m):
            xi = X_shuffled[i]
            yi = y_shuffled[i]

            # Compute prediction and error for one example
            prediction = hypothesis(xi, theta0, theta1)
            error = prediction - yi

            # Update parameters immediately
            theta0 -= alpha * error
            theta1 -= alpha * error * xi

        # Compute total cost after each full pass (epoch)
        cost_history.append(computeCost(X, y, theta0, theta1))

    return theta0, theta1, cost_history


# -----------------------------
# STEP 5: TRAINING COMPARISON
# -----------------------------

theta0, theta1 = 0, 0
alpha = 0.0000001
iterations = 1000

# Run each version
theta0_b, theta1_b, cost_b = batch_gradient_descent(
    X, y, theta0, theta1, alpha, iterations
)
theta0_s, theta1_s, cost_s = stochastic_gradient_descent(
    X, y, theta0, theta1, alpha, 100
)

# -----------------------------
# STEP 6: RESULTS AND VISUALIZATION
# -----------------------------

print("Batch GD → theta0:", theta0_b, "theta1:", theta1_b)
print("Stochastic GD → theta0:", theta0_s, "theta1:", theta1_s)

# Plot the data and regression lines
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, hypothesis(X, theta0_b, theta1_b), color="red", label="Batch GD")
plt.plot(
    X,
    hypothesis(X, theta0_s, theta1_s),
    color="green",
    linestyle="--",
    label="Stochastic GD",
)
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price ($)")
plt.title("Comparison of Gradient Descent Methods")
plt.legend()
plt.show()

# Plot cost function convergence
plt.plot(cost_b, label="Batch GD", color="red")
plt.plot(cost_s, label="Stochastic GD", color="green")
plt.xlabel("Iterations / Epochs")
plt.ylabel("Cost")
plt.title("Cost Function Reduction Comparison")
plt.legend()
plt.show()
