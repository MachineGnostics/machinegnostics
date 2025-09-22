'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar
'''

import numpy as np
from machinegnostics.magcal import DataConversion, GnosticsWeights

def __gcorrelation(data_1:np.ndarray, data_2:np.ndarray) -> float:
    """
    Calculate the Gnostic correlation between two data samples using robust irrelevance-based weighting.

    This function implements the robust gnostic correlation as described in Kovanic & Humber (2015).
    The method uses irrelevance functions to construct weights,
    providing a robust alternative to classical Pearson correlation. It is less sensitive to outliers,
    does not assume normality.

    
    """
    # if len(data_1) != len(data_2):
    #     raise ValueError("Input arrays must have the same length.")
    # if len(data_1) == 0 or len(data_2) == 0:
    #     raise ValueError("Input arrays must not be empty.")
    # if not isinstance(data_1, np.ndarray) or not isinstance(data_2, np.ndarray):
    #     raise ValueError("Input arrays must be numpy arrays.")
    
    zx = data_1 / np.mean(data_1)
    zy = data_2 / np.mean(data_2)

    dc = DataConversion()
    x_norm = dc._convert_az(zx)
    y_norm = dc._convert_az(zy)

    gwx = GnosticsWeights()
    wx = gwx._get_gnostic_weights(x_norm)
    gwy = GnosticsWeights()
    wy = gwy._get_gnostic_weights(y_norm)

    W = np.sqrt(wx * wy)

    numerator = np.sum(x_norm * W * W * y_norm)
    denominator = np.sqrt(np.sum(x_norm * W * W * x_norm) * np.sum(y_norm * W * W * y_norm))
    if denominator == 0:
        return 0.0
    return numerator / denominator

# def gcorrelation(data_1: np.ndarray, data_2: np.ndarray) -> np.ndarray:
#     if data_1.ndim == 1:
#         data_1 = data_1[np.newaxis, :]
#     if data_2.ndim == 1:
#         data_2 = data_2[np.newaxis, :]
#     if data_1.shape[1] != data_2.shape[1]:
#         raise ValueError("Each row in data_1 and data_2 must have the same number of samples (columns).")

#     n_x, n_samples = data_1.shape
#     n_y = data_2.shape[0]
#     corr_matrix = np.zeros((n_x, n_y))

#     for i in range(n_x):
#         for j in range(n_y):
#             corr_matrix[i, j] = __gcorrelation(data_1[i], data_2[j])
#     return corr_matrix