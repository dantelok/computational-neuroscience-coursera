import numpy as np

# Define matrices W, u, and M
W = np.array([
    [0.6, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.6, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.6, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.6, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.6]
])

u = np.array([0.6, 0.5, 0.6, 0.2, 0.1])

x = 0.75
M = np.array([
    [-x, 0, x, x, 0],
    [0, -x, 0, x, x],
    [x, 0, -x, 0, x],
    [x, x, 0.0, -x, 0],
    [0, x, x, 0, -x]
])

# The steady-state output satisfies: v_ss = F(Wu + Mv_ss)
# If F is the identity (linear), then rearrange:
# (I - M) v_ss = W u
# v_ss = (I - M)^(-1) W u

I = np.eye(5)
v_ss = np.linalg.inv(I - M).dot(W.dot(u))
print(v_ss)
