import numpy as np

def create_windows(X, y, window_size=8, window_step=1, window_shift=1):

    X_windows, y_windows = [], []
    idx_windows, idx_y = [], []

    X_np = X.values if hasattr(X, "values") else X
    y_np = y.values if hasattr(y, "values") else y

    index_X = X.index if hasattr(X, "index") else np.arange(len(X_np))
    index_y = y.index if hasattr(y, "index") else np.arange(len(y_np))

    n_samples = min(len(X_np), len(y_np))

    for i in range(0, n_samples - (window_size - 1) * window_step, window_shift):
        indices = [i + j * window_step for j in range(window_size)]

        X_windows.append(X_np[indices])
        y_windows.append(y_np[indices[-1]])

        # salva índices originais
        idx_windows.append(index_X[indices])
        idx_y.append(index_y[indices[-1]])

    return (
        np.array(X_windows, dtype=np.float32),
        np.array(y_windows, dtype=np.float32),
        np.array(idx_windows), 
        np.array(idx_y),       
    )