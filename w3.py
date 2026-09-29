import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Same data-generating process
_, _, true_coef = make_regression(n_samples=200, n_features=5, n_informative=3,
                                  noise=20, coef=True, random_state=42)

coefs, errors = [], []
for seed in range(100):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 5))
    y = X @ true_coef + rng.normal(scale=20, size=200)
    m = LinearRegression().fit(X, y)
    coefs.append(m.coef_)
    errors.append(mean_squared_error(y, m.predict(X)))

coefs = np.array(coefs)
errors = np.array(errors)
print("true:", true_coef.round(2))
print("mean:", coefs.mean(0).round(2), "std:", coefs.std(0).round(2))
print("MSE mean:", errors.mean().round(1), "MSE std:", errors.std().round(1))

# Plot
names = [f"X{i+1}" for i in range(coefs.shape[1])]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

axes[0].boxplot(coefs, tick_labels=names)
axes[0].scatter(range(1, len(names) + 1), true_coef, color="red", zorder=3, label="True value")
axes[0].set_title("Estimated coefficients across 100 datasets")
axes[0].set_ylabel("Coefficient")
axes[0].legend()

axes[1].boxplot(coefs - true_coef, tick_labels=names)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("Estimation error (estimate - true)")
axes[1].set_ylabel("Deviation from true coefficient")

axes[2].hist(errors, bins=15, color="steelblue", edgecolor="white")
axes[2].axvline(errors.mean(), color="red", linestyle="--", label=f"Mean = {errors.mean():.1f}")
axes[2].set_title("Model error (MSE) across datasets")
axes[2].set_xlabel("MSE")
axes[2].legend()

plt.suptitle("Experiment 1: Same Model, Different Samples", fontsize=14)
plt.tight_layout()
plt.savefig("/Users/jiahuijiang/w3_experiment1.png", dpi=150)
            