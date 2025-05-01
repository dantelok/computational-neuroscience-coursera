import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Parameters for the two Gaussian distributions
mu1, sigma1 = 5, 0.5  # For stimulus s1
mu2, sigma2 = 7, 1    # For stimulus s2

# Asymmetric costs
cost_s1_given_s2 = 1  # Cost of guessing s1 when it was s2
cost_s2_given_s1 = 2  # Cost of guessing s2 when it was s1

# Equal priors
p_s1 = 0.5
p_s2 = 0.5

# Decision threshold for likelihood ratio
threshold_ratio = (cost_s1_given_s2 * p_s2) / (cost_s2_given_s1 * p_s1)

# Define range of r values
r_vals = np.linspace(3, 9, 1000)

# Compute the likelihoods
p_r_given_s1 = norm.pdf(r_vals, mu1, sigma1)
p_r_given_s2 = norm.pdf(r_vals, mu2, sigma2)

# Compute the likelihood ratio
likelihood_ratio = p_r_given_s1 / p_r_given_s2

# Find the point where the likelihood ratio is closest to the threshold
idx = np.argmin(np.abs(likelihood_ratio - threshold_ratio))
decision_threshold = r_vals[idx]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(r_vals, likelihood_ratio, label=r"$\frac{P(r|s_1)}{P(r|s_2)}$")
plt.axhline(threshold_ratio, color='red', linestyle='--', label='Cost Ratio Threshold = 0.5')
plt.axvline(decision_threshold, color='green', linestyle='--', label=f'Threshold r ≈ {decision_threshold:.2f}')
plt.title('Likelihood Ratio with Asymmetric Costs')
plt.xlabel('Firing Rate (r)')
plt.ylabel('Likelihood Ratio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(decision_threshold)
