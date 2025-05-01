import numpy as np


def find_principal_eigenvector(Q):
    """
    Given a correlation matrix Q, compute the principal eigenvector
    corresponding to the largest eigenvalue (Hebb learning long-term solution).

    Args:
        Q (np.ndarray): 2x2 correlation matrix

    Returns:
        np.ndarray: Principal eigenvector (normalized)
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(Q)

    # Find the index of the largest eigenvalue
    max_index = np.argmax(eigenvalues)

    # Extract the corresponding eigenvector
    principal_eigenvector = eigenvectors[:, max_index]

    # Normalize the eigenvector for consistency
    principal_eigenvector = principal_eigenvector / np.linalg.norm(principal_eigenvector)

    return principal_eigenvector


Q = np.array([[0.2, 0.1],
              [0.1, 0.15]])

principal_eigenvector = find_principal_eigenvector(Q)
print("Principal Eigenvector (Final w):", principal_eigenvector * 2)
