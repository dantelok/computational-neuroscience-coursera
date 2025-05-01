import numpy as np
import matplotlib.pyplot as plt
import pickle

# 1. Load the data
with open('c10p1.pickle', 'rb') as f:
    data = pickle.load(f)

data = data['c10p1']
# 2. Zero-mean the data
data = np.array(data)
data_mean = np.mean(data, axis=0)
data_centered = data - data_mean

# 3. Plot the zero-meaned data
plt.figure(figsize=(6, 6))
plt.plot(data_centered[:, 0], data_centered[:, 1], 'o')
plt.title('Zero-Mean Centered Data')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
plt.show()

# 4. Initialize parameters
eta = 1  # η = 1
alpha = 1  # α = 1
dt = 0.01  # Δt = 0.01
num_iters = 100000  # 100k iterations
N = data_centered.shape[0]  # Number of data points

# Random initial weight vector w (2D)
w = np.random.randn(2)

# 5. Online Learning using Oja's Rule
for iter in range(num_iters):
    # Pick data point cyclically
    u = data_centered[iter % N]

    # Compute v
    v = np.dot(w, u)

    # Update w
    w += dt * eta * (v * u - alpha * v ** 2 * w)

# 6. Display the final weight vector
print('Learned weight vector w:', w)

# 7. Plot the data and the final learned direction
plt.figure(figsize=(6, 6))
plt.plot(data_centered[:, 0], data_centered[:, 1], 'o', label='Data')
plt.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='r', label='Learned w', width=0.01)
plt.title('Learned Direction Over Zero-Mean Data')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()


# ------------------------------------------------------------------
# Mean shift
# 1. Add a mean shift
mean_shift = np.array([2.0, -1.5])  # example shift: move +2 in x, -1.5 in y
data_shifted = data + mean_shift

# 2. Plot the shifted data
plt.figure(figsize=(6, 6))
plt.plot(data_shifted[:, 0], data_shifted[:, 1], 'o')
plt.title('Shifted Data')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
plt.show()

# 3. Setup for Oja's learning
eta = 1
alpha = 1
dt = 0.01
num_iters = 100000
N = data_shifted.shape[0]

w = np.random.randn(2)  # Random initialization

# 4. Online Oja's Learning on shifted data
for iter in range(num_iters):
    u = data_shifted[iter % N]
    v = np.dot(w, u)
    w += dt * eta * (v * u - alpha * v ** 2 * w)

print('Learned weight vector (shifted data):', w)

# 5. Plot final learned direction
plt.figure(figsize=(6, 6))
plt.plot(data_shifted[:, 0], data_shifted[:, 1], 'o', label='Shifted Data')
plt.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='r', label='Learned w', width=0.01)
plt.title('Learned Direction After Mean Shift')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

# ------------------------------------------------------------------
# Oja's rule vs Hebb's rule comparison

# Reset and center the original data
data = np.array(data)
data_mean = np.mean(data, axis=0)
data_centered = data - data_mean
N = data_centered.shape[0]

# Parameters
eta = 1
alpha = 1
dt = 0.01
num_iters = 10000

# Initialize two weight vectors
w_oja = np.random.randn(2)
w_hebb = np.copy(w_oja)  # Same start for fair comparison

# Lists to store norms over time
norms_oja = []
norms_hebb = []

# Training loop
for iter in range(num_iters):
    u = data_centered[iter % N]

    v_oja = np.dot(w_oja, u)
    w_oja += dt * eta * (v_oja * u - alpha * v_oja ** 2 * w_oja)
    norms_oja.append(np.linalg.norm(w_oja))

    v_hebb = np.dot(w_hebb, u)
    w_hebb += dt * eta * (v_hebb * u)  # Pure Hebbian rule (NO decay term)
    norms_hebb.append(np.linalg.norm(w_hebb))

# Plot norms
plt.figure(figsize=(8, 5))
plt.plot(norms_oja, label='Oja\'s Rule (Normalized)')
plt.plot(norms_hebb, label='Hebb\'s Rule (Exploding)', linestyle='--')
plt.xlabel('Iterations')
plt.ylabel('Norm of w')
plt.title('Norm Comparison: Oja\'s vs Hebb\'s Learning')
plt.legend()
plt.grid(True)
plt.show()
