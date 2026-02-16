'''
Entropy - Machine Gnostics Framework
- Estimating and Quantifying geometries

Machine Gnostics

Author: Nirmal Parmar
'''

import logging
from typing import Union, Optional
import numpy as np

from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.magcal import EGDF, QGDF
from machinegnostics.magcal.util.narwhals_df import narwhalify


@narwhalify
def entropy(data: np.ndarray, 
            data_compare: Optional[np.ndarray] = None,
            S: Union[float, str] = 'auto', 
            case: str = 'i',
            z0_optimize: bool = False, 
            data_form: str = 'a',
            tolerance: float = 1e-6,
            verbose: bool = False) -> float:
    """
    Calculates the Gnostic Entropy of a data array or the difference between two arrays with the help of EGDF/QGDF.

    This metric evaluates uncertainty or disorder using the Machine Gnostics framework.
    If only `data` is provided, it calculates the entropy of that distribution.
    If `data_compare` is also provided, it calculates the entropy of the residuals 
    (data_compare - data).

    Parameters
    ----------
    data : array-like or dataframe/series
        Reference data values (e.g., Ground Truth) or the single dataset to evaluate. 
        Must be a 1D array.
    data_compare : array-like or dataframe/series, optional
        Data values to compare against the reference (e.g., Estimated/Predicted values). 
        If provided, entropy is calculated on (data_compare - data).
        If None, entropy is calculated on `data` directly.
    S : float or 'auto', default='auto'
        Scale parameter for the Gnostic Distribution Function. 
        If float, suggested range is [0.01, 2].
    case : {'i', 'j'}, default='i'
        The type of variance geometry to use:
        - 'i': Estimating geometry (leads to Entropy = 1 - mean(fi)). Used for standard uncertainty estimation.
        - 'j': Quantifying geometry (leads to Entropy = mean(fj) - 1). Used for quantifying outliers or extreme deviations.
    z0_optimize : bool, default=True
        Whether to optimize the location parameter z0.
    data_form : {'a', 'm'}, default='a'
        The nature of the data relationship:
        - 'a': Additive (diff = data_compare - data). Standard for most regression errors.
        - 'm': Multiplicative.
    tolerance : float, default=1e-6
        Convergence tolerance for the optimization numerical methods.
    verbose : bool, default=False
        If True, prints detailed calculation logs.

    Returns
    -------
    float
        The calculated gnostic entropy value. 
        For case 'i', values are typically in [0, 1], where 0 indicates perfect certainty.

    Raises
    ------
    ValueError
        If inputs have mismatched shapes, invalid dimensions, contain NaNs/Infs, or valid options are not selected.
    TypeError
        If inputs define incorrect data types.
    """ 
    logger = get_logger('entropy', level=logging.WARNING if not verbose else logging.INFO)
    logger.info("Calculating gnostic entropy...")

    # --- Input Validation ---
    
    # 1. Validate 'data'
    if not isinstance(data, (list, tuple, np.ndarray)):
        msg = "data must be array-like (list, tuple, or numpy array)."
        logger.error(msg)
        raise TypeError(msg)

    data = np.asarray(data).flatten()
    
    if data.size == 0:
        msg = "Input array 'data' must not be empty."
        logger.error(msg)
        raise ValueError(msg)

    if not np.all(np.isfinite(data)):
        msg = "Input 'data' must not contain NaN or Inf values."
        logger.error(msg)
        raise ValueError(msg)
    
    if data.size == 1:
        return 0.0

    # 2. Validate 'data_compare' if provided
    y_diff = None
    
    if data_compare is not None:
        if not isinstance(data_compare, (list, tuple, np.ndarray)):
            msg = "data_compare must be array-like (list, tuple, or numpy array)."
            logger.error(msg)
            raise TypeError(msg)
            
        data_compare = np.asarray(data_compare).flatten()

        if data.shape != data_compare.shape:
            msg = f"Shape mismatch: data {data.shape} and data_compare {data_compare.shape} must have the same shape."
            logger.error(msg)
            raise ValueError(msg)

        if not np.all(np.isfinite(data_compare)):
            msg = "Input 'data_compare' must not contain NaN or Inf values."
            logger.error(msg)
            raise ValueError(msg)
            
        # Case A: Two arrays provided -> Calculate Difference
        y_diff = data_compare - data
    else:
        # Case B: One array provided -> Use directly
        y_diff = data

    # 3. Argument Validation
    if isinstance(S, str):
        if S != 'auto':
            raise ValueError("S must be a float or 'auto'.")
    elif isinstance(S, (int, float)):
         if S < 0.01 or S >= 2:
            logger.warning("S is outside the suggested range [0.01, 2].")
    else:
        raise TypeError("S must be a float or 'auto'.")

    if data_form not in ['a', 'm']:
        raise ValueError("data_form must be 'a' (additive) or 'm' (multiplicative).")
    
    if case not in ['i', 'j']:
        raise ValueError("case must be 'i' (estimating) or 'j' (quantifying).")

    # --- Calculation ---

    # GDF processing per case
    if case == 'i':
        logger.info("Using estimating geometry (EGDF) for entropy calculation.")
        gdf = EGDF(verbose=verbose,
                   z0_optimize=z0_optimize,
                   S=S,
                   data_form=data_form,
                   tolerance=tolerance,
                   flush=False,
                   n_points=50)
        gdf.fit(data=y_diff)
        # Entropy for estimating case: 1 - expected belief
        gnostic_entropy = 1.0 - gdf.fi.mean()

    else:
        logger.info("Using quantifying geometry (QGDF) for entropy calculation.")
        gdf = QGDF(verbose=verbose,
                   z0_optimize=z0_optimize,
                   S=S,
                   data_form=data_form,
                   tolerance=tolerance,
                   flush=False,
                   n_points=50)
        gdf.fit(data=y_diff)
        # Entropy for quantifying case: expected weight - 1
        gnostic_entropy = gdf.fj.mean() - 1.0

    logger.info(f"Gnostic entropy calculated: {gnostic_entropy:.6f}")
    return float(gnostic_entropy)