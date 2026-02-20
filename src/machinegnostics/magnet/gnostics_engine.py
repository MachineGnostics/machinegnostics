"""Gnostic modifier engine for Magnet.

Developer notes
---------------
This engine intentionally handles *only* geometry/GDF-based weight modifiers.
Residual-to-gnostic-weight computation is centralized in `GnosticLoss` so the
project has a single source of truth for sample-wise gnostic weighting.
"""
import logging
from machinegnostics.magcal import (EGDF, QGDF, ELDF, QLDF)
from machinegnostics.magcal.util.logging import get_logger
import numpy as np

class GnosticEngine:
    def __init__(self, 
                 geometry: str = 'E', 
                 gdf: str | None = 'global', 
                 verbose: bool = False, S: float | int | str = 'auto'):
        # Normalize inputs
        geom = (geometry or 'E').upper()
        if geom not in ('E', 'Q'):
            raise ValueError("geometry must be 'E' or 'Q'")
        gdf_norm = None
        if isinstance(gdf, str):
            s = gdf.lower()
            # accept aliases: 'egdf','eldf','qgdf','qldf'
            if s in ('global', 'egdf', 'qgdf'):
                gdf_norm = 'global'
            elif s in ('local', 'eldf', 'qldf'):
                gdf_norm = 'local'
            elif s in ('none', 'null'):
                gdf_norm = None
            else:
                # if explicit alias encodes geometry, keep geom consistent
                gdf_norm = s if s in ('global', 'local') else None
        elif gdf is None:
            gdf_norm = None
        else:
            raise ValueError("gdf must be 'global', 'local', alias ('egdf','eldf','qgdf','qldf'), or None")
        self.geometry = geom
        self.gdf = gdf_norm
        self.S = S
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)

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
