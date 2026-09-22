import numpy as np
import matplotlib.pyplot as plt


# 1. Generate nonlinear data
def make_data(n=80, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    y = 3 * np.sin(x) + 0.3 * x + rng.normal(0, 0.6, n)
    return x, y


# 2. Add outliers to 8% of the data
def add_outliers(y, seed=2):
    rng = np.random.default_rng(seed)
    y = y.copy()

    n_outliers = int(len(y) * 0.08)
    idx = rng.choice(len(y), n_outliers, replace=False)

    y[idx] += rng.choice([-8, 8], n_outliers)

    return y


# 3. Split data: 70% training, 30% testing
def split_data(x, y, seed=3):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))

    n_test = int(len(x) * 0.3)

    test = idx[:n_test]
    train = idx[n_test:]

    return x[train], y[train], x[test], y[test]


# 4. Fit polynomial regression using least squares
def fit_poly(x, y, degree):
    X = np.vander(x, degree + 1, increasing=True)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs


# 5. Make predictions
def predict(coeffs, x):
    X = np.vander(x, len(coeffs), increasing=True)
    return X @ coeffs


# 6. Evaluation metrics
def mse(y, pred):
    return np.mean((y - pred) ** 2)


def mae(y, pred):
    return np.mean(np.abs(y - pred))


# ------------------------------------------------
# Main experiment
# ------------------------------------------------

degrees = range(1, 11)

# Generate clean data
x, y = make_data()

# Add outliers
y = add_outliers(y)

# Train/test split
x_train, y_train, x_test, y_test = split_data(x, y)

mse_results = []
mae_results = []


# Try polynomial degrees 1 through 10
for degree in degrees:

    coeffs = fit_poly(x_train, y_train, degree)
    pred = predict(coeffs, x_test)

    mse_results.append(mse(y_test, pred))
    mae_results.append(mae(y_test, pred))


# Find the best degree according to each metric
best_mse_degree = np.argmin(mse_results) + 1
best_mae_degree = np.argmin(mae_results) + 1

print("Best degree by MSE:", best_mse_degree)
print("Best degree by MAE:", best_mae_degree)


# ------------------------------------------------
# Plot the results
# ------------------------------------------------

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

ax1.plot(degrees, mse_results, marker="o", label="MSE")
ax2.plot(degrees, mae_results, marker="s", label="MAE")

ax1.set_xlabel("Polynomial Degree")
ax1.set_ylabel("Test MSE")
ax2.set_ylabel("Test MAE")

plt.title("MSE vs MAE Model Selection with Outliers")

plt.tight_layout()
plt.savefig("mse_vs_mae.png")
plt.show()