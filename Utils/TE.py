import numpy as np

def generate_lagged_vectors(signal: np.ndarray, lag: int) -> np.ndarray :

    """Generate lagged vectors from a signal."""

    n = len(signal)
    lagged_vectors = np.array([signal[i: n - lag + i + 1] for i in range(lag)]).T
    
    return lagged_vectors