import numpy as np
def one_hot(labels, num_classes):
    out = np.zeros((len(labels), num_classes))
    out[np.arange(len(labels)), labels] = 1.0
    return out

def train_test_split(x, y, test_size=0.2, seed=None):
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]