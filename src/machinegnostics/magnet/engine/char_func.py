import numpy as np

def _calculate_q(Z: np.ndarray, Z0: np.ndarray, S: float) -> np.ndarray:
    """
    Calculate the q parameter for gnostic calculations.
    
    q = (Z / Z0)^(1/S)
    
    Parameters
    ----------
    Z : ndarray
        Values to normalize
    Z0 : ndarray or float
        Reference/ideal value
    S : float
        Scale parameter
        
    Returns
    -------
    ndarray
        q parameter values
    """
    epsilon = 1e-6
    Z0 = np.where(Z0 == 0, epsilon, Z0)
    return np.power(Z / Z0, 1.0 / S)

def _gnostic_fidelity(Z: np.ndarray, Z0: np.ndarray, S: float) -> np.ndarray:
    """
    Calculate gnostic fidelity (agreement measure).
    
    fi = 2 / (q^2 + q^-2)
    
    Parameters
    ----------
    Z : ndarray
        Values to evaluate
    Z0 : ndarray or float
        Reference value
    S : float
        Scale parameter
        
    Returns
    -------
    ndarray
        Fidelity values in range [0, 1]
    """
    q = _calculate_q(Z, Z0, S)
    q_sq = np.power(q, 2)
    q_inv_sq = np.power(q, -2)
    fi = 2.0 / (q_sq + q_inv_sq)
    fi = np.where(np.isnan(fi), 0, fi)
    return fi

def _gnostic_infidelity(Z: np.ndarray, Z0: np.ndarray, S: float) -> np.ndarray:
    """
    Calculate gnostic un-fidelity (1/fidelity).
    
    fj = 1 / fi (capped at 1e12 to avoid overflow)
    
    Parameters
    ----------
    Z : ndarray
        Values to evaluate
    Z0 : ndarray or float
        Reference value
    S : float
        Scale parameter
        
    Returns
    -------
    ndarray
        Un-fidelity values
    """
    fi = _gnostic_fidelity(Z, Z0, S)
    fi = np.where(fi == 0, 1e-6, fi)
    fj = 1.0 / fi
    fj = np.where(fj > 1e12, 1e12, fj)
    return fj

def _gnostic_irrelevance(Z: np.ndarray, Z0: np.ndarray, S: float) -> np.ndarray:
    """
    Calculate gnostic irrelevance (deviation measure).
    
    h = (q^-2 - q^2) / (q^2 + q^-2)
    
    Parameters
    ----------
    Z : ndarray
        Values to evaluate
    Z0 : ndarray or float
        Reference value
    S : float
        Scale parameter
        
    Returns
    -------
    ndarray
        Irrelevance values in range [-1, 1]
    """
    q = _calculate_q(Z, Z0, S)
    q_sq = np.power(q, 2)
    q_inv_sq = np.power(q, -2)
    h = (q_inv_sq - q_sq) / (q_sq + q_inv_sq)
    h = np.where(np.isnan(h), 0, h)
    return h

def _prim_gnostic_fidelity(Z: np.ndarray, Z0: np.ndarray, S: float) -> np.ndarray:
    """
    Derivative of fidelity w.r.t. output.
    
    ∂f/∂Z = -2/S * f * h
    
    Parameters
    ----------
    Z : ndarray
        Values to evaluate
    Z0 : ndarray or float
        Reference value
    S : float
        Scale parameter
        
    Returns
    -------
    ndarray
        Derivatives of fidelity
    """
    fi = _gnostic_fidelity(Z, Z0, S)
    h = _gnostic_irrelevance(Z, Z0, S)
    prim = -(2.0 / S) * fi * h
    return prim

def _prim_gnostic_infidelity(Z: np.ndarray, Z0: np.ndarray, S: float) -> np.ndarray:
    """
    Derivative of un-fidelity (1/f) w.r.t. output.
    
    ∂(1/f)/∂Z = -2/S * (1/f) * h
    
    This is the Gnostic Backpropagation error derivative.
    
    Parameters
    ----------
    Z : ndarray
        Values to evaluate
    Z0 : ndarray or float
        Reference value
    S : float
        Scale parameter
        
    Returns
    -------
    ndarray
        Derivatives of un-fidelity (GBP error signal)
    """
    fi = _gnostic_fidelity(Z, Z0, S)
    fi = np.where(fi == 0, 1e-6, fi)
    h = _gnostic_irrelevance(Z, Z0, S)
    prim_infi = -(2.0 / S) * (1.0 / fi) * h
    return prim_infi