# ----------------------------------------
# STEP 1: DATA PREPARATION + NORMALIZATION
# ----------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Suppose we have the following housing dataset:
# Features: [Size (sqft), Rooms, Age (years)]
# Target: Price ($)
X = np.array(
    [
        [850, 2, 10],
        [900, 3, 5],
        [1200, 3, 15],
        [1500, 4, 8],
        [2000, 4, 12],
        [2200, 5, 20],
        [2500, 5, 7],
        [3000, 6, 5],
    ]
)

y = np.array([150000, 170000, 200000, 230000, 260000, 280000, 300000, 340000])

# Number of samples
m = X.shape[0]

# --- Feature Scaling (Z-score normalization) ---
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = (X - mean) / std

# Add bias term (column of 1s)
X_bias = np.c_[np.ones((m, 1)), X_norm]  # shape: (m, n+1)

print("Mean of features:", mean)
print("Standard deviation of features:", std)
print("Normalized feature matrix:\n", X_bias[:5])  # print first 5 rows

# ----------------------------------------
# STEP 2: VECTORIZED HYPOTHESIS FUNCTION
# ----------------------------------------


def hypothesis(X, theta):
    """
    Computes h = X * theta
    X: input matrix (m x n+1)
    theta: parameters vector (n+1 x 1)
    """
    return np.dot(X, theta)


# ----------------------------------------
# STEP 3: VECTORIZED COST FUNCTION
# ----------------------------------------


def computeCost(X, y, theta):
    """
    J(theta) = (1/2m) * (Xθ - y)^T (Xθ - y)
    """
    m = len(y)
    predictions = hypothesis(X, theta)
    error = predictions - y
    cost = (1 / (2 * m)) * np.dot(error.T, error)
    return cost


# ----------------------------------------
# STEP 4: VECTORIZED GRADIENT DESCENT
# ----------------------------------------


def gradientDescent(X, y, theta, alpha, iterations):
    """
    Performs gradient descent to minimize cost J(theta)
    """
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        predictions = hypothesis(X, theta)
        error = predictions - y
        gradient = (1 / m) * np.dot(X.T, error)
        theta = theta - alpha * gradient
        cost_history.append(computeCost(X, y, theta))

    return theta, cost_history


# ----------------------------------------
# STEP 5: EXPERIMENTATION
# ----------------------------------------

# Initialize theta (n+1 parameters)
theta = np.zeros(X_bias.shape[1])

# Try different learning rates
alphas = [0.01, 0.03, 0.1]
iterations = 500

plt.figure(figsize=(8, 5))

for alpha in alphas:
    theta_init = np.zeros(X_bias.shape[1])
    theta_final, cost_history = gradientDescent(
        X_bias, y, theta_init, alpha, iterations
    )
    plt.plot(range(iterations), cost_history, label=f"alpha={alpha}")
    print(f"Alpha={alpha} => Final cost={cost_history[-1]:.2f}, Thetas={theta_final}")

plt.xlabel("Iterations")
plt.ylabel("Cost J(theta)")
plt.title("Convergence Speed for Different Learning Rates")
plt.legend()
plt.show()

# ----------------------------------------
# STEP 6: COMPARISON WITH SCIKIT-LEARN
# ----------------------------------------

# Train using scikit-learn's LinearRegression
model = LinearRegression()
model.fit(X, y)

print("\nScikit-learn coefficients:", model.coef_)
print("Scikit-learn intercept:", model.intercept_)

# Compare predictions
y_pred_sklearn = model.predict(X)
y_pred_gd = hypothesis(X_bias, theta_final)

print("\nManual GD vs Sklearn predictions:")
for i in range(len(y)):
    print(
        f"House {i+1}: GD={y_pred_gd[i]:.1f}, Sklearn={y_pred_sklearn[i]:.1f}, Actual={y[i]:.1f}"
    )

# Visualization (if 2D features, we could plot a plane; here just show cost)
print("\nFinal GD Cost:", computeCost(X_bias, y, theta_final))
