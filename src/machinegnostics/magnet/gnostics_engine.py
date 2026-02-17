"""
Gnostic integration for neural networks: weights updates, geometry (E/Q), GDF (local/global).

geometry semantics:
- 'E' → Estimating (Euclidian)
- 'Q' → Quantification (Minkowskian)
"""
import numpy as np
import logging
from typing import Tuple
from machinegnostics.magcal import (GnosticsWeights, EGDF, QGDF, ELDF, QLDF)
from machinegnostics.magcal.util.logging import get_logger

class GnosticEngine:
    def __init__(self, geometry: str = 'E', gdf: str = 'global', verbose: bool = False, S: float | int | str = 'auto'):
        if geometry not in ('E', 'Q'):
            raise ValueError("geometry must be 'E' or 'Q'")
        if gdf not in ('global', 'local', None):
            raise ValueError("gdf must be 'global', 'local', or None")
        self.geometry = geometry
        self.gdf = gdf
        self.S = S
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)

    def compute_sample_weights(self, residuals: np.ndarray) -> np.ndarray:
        gwc = GnosticsWeights()
        gw = gwc._get_gnostic_weights(residuals)
        # Ensure 1D shape aligned to residuals length
        try:
            gw = np.asarray(gw)
            if gw.ndim > 1:
                # Prefer reducing the last axis if it matches len(residuals)
                if gw.shape[-1] == residuals.shape[0]:
                    gw = gw.mean(axis=tuple(range(gw.ndim - 1)))
                else:
                    gw = gw.mean(axis=-1)
            gw = gw.reshape(-1)
            if gw.size != residuals.shape[0]:
                # Fallback: inverse-residual weighting (robust to outliers)
                r = np.abs(residuals).reshape(-1)
                eps = 1e-8
                gw = 1.0 / (r + eps)
        except Exception:
            # Fallback weights: uniform
            gw = np.ones(residuals.shape[0])
        # normalize
        total = float(np.sum(gw))
        if total <= 0:
            return np.ones(residuals.shape[0]) / residuals.shape[0]
        return gw / total

    def apply_gdf_modifier(self, z: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # Attempt to apply geometry + GDF weight modifiers if available.
        self.logger.info(f"Applying GDF modifier: geometry={self.geometry}, gdf={self.gdf}")
        # user warning for more than 100 len z
        if len(z) > 300:
             self.logger.warning(f"Data is large for GDF calculation and can be time consuming! Continue by switching 'gdf=None'.")
        try:
            if self.geometry == 'E' and self.gdf == 'global':
                gdf = EGDF(flush=False, n_points=100, z0_optimize=False, S=self.S)
            elif self.geometry == 'E' and self.gdf == 'local':
                gdf = ELDF(flush=False, n_points=100, z0_optimize=False, S=self.S)
            elif self.geometry == 'Q' and self.gdf == 'global':
                gdf = QGDF(flush=False, n_points=100, z0_optimize=False, S=self.S)
            else:
                gdf = QLDF(flush=False, n_points=100, z0_optimize=False, S=self.S)
            # Try common method names safely
            gdf.fit(z)
            if hasattr(gdf, 'fj'):
                mod = np.asarray(gdf.fj) ** 2
                # clip to a reasonable maximum
                mod = np.clip(mod, 1, 1e12)
            else:
                mod = np.ones_like(weights)
            # Align modifier shape to sample count
            m = mod.reshape(-1)
            if m.size != z.shape[0]:
                # If modifier length mismatches, fall back to original weights
                m = np.ones(z.shape[0])
            s = float(np.sum(m))
            return (m / s) if s > 0 else weights
        except Exception:
            # Fallback: return original weights
            return weights
