from abc import ABCMeta, abstractmethod

import numpy as np
from machinegnostics.magcal import ELDF, EGDF, QLDF, QGDF


class BaseInitializer(metaclass=ABCMeta):
    @abstractmethod
    def initialize(self, x):
        pass


class RandomNormal(BaseInitializer):
    def __init__(self, mean=0., std=1.):
        self._mean = mean
        self._std = std

    def initialize(self, x):
        x[:] = np.random.normal(loc=self._mean, scale=self._std, size=x.shape)


class RandomUniform(BaseInitializer):
    def __init__(self, low=0., high=1.):
        self._low = low
        self._high = high

    def initialize(self, x):
        x[:] = np.random.uniform(self._low, self._high, size=x.shape)


class Zeros(BaseInitializer):
    def initialize(self, x):
        x[:] = np.zeros_like(x)


class Ones(BaseInitializer):
    def initialize(self, x):
        x[:] = np.ones_like(x)


class TruncatedNormal(BaseInitializer):
    def __init__(self, mean=0., std=1.):
        self._mean = mean
        self._std = std

    def initialize(self, x):
        x[:] = np.random.normal(loc=self._mean, scale=self._std, size=x.shape)
        truncated = 2*self._std + self._mean
        x[:] = np.clip(x, -truncated, truncated)


class Constant(BaseInitializer):
    def __init__(self, v):
        self._v = v

    def initialize(self, x):
        x[:] = np.full_like(x, self._v)

# gnostic weights
class Gnostic(BaseInitializer):
    """Data-driven gnostic initializer.

    Developer notes
    ---------------
    This initializer estimates a GDF model and maps its learned distribution
    characteristics to parameter initialization weights.

    Unlike Gaussian initializers, this class can be driven by provided data
    (`data`) to reduce dependency on synthetic normal sampling.
    """

    def __init__(self,
                 gdf: str,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S: str | float = 'auto',
                 vars: bool | None = None,
                 varS: bool = False,
                 minimum_varS: float = 0.1,
                 z0_optimize: bool = True,
                 tolerance: float = 1e-9,
                 data_form: str = 'a',
                 n_points: int = 1000,
                 homogeneous: bool = True,
                 catch: bool = True,
                 weights: np.ndarray = None,
                 wedf: bool = False,
                 opt_method: str = 'Powell',
                 verbose: bool = False,
                 max_data_size: int = 1000,
                 flush: bool = True,
                 data: np.ndarray = None):
        self.gdf = (gdf or '').lower()
        if self.gdf not in ('eldf', 'egdf', 'qldf', 'qgdf'):
            raise ValueError("gdf must be one of: 'eldf', 'egdf', 'qldf', 'qgdf'")

        self.DLB = DLB
        self.DUB = DUB
        self.LB = LB
        self.UB = UB
        self.S = S
        # accept both `varS` and legacy typo/name `vars`
        self.varS = bool(varS if vars is None else vars)
        self.minimum_varS = minimum_varS
        self.z0_optimize = z0_optimize
        self.tolerance = tolerance
        self.data_form = data_form
        self.n_points = int(n_points)
        self.homogeneous = homogeneous
        self.catch = catch
        self.weights = weights
        self.wedf = wedf
        self.opt_method = opt_method
        self.verbose = verbose
        self.max_data_size = int(max_data_size)
        self.flush = flush
        self.data = data

    def _build_gdf_model(self):
        cls_map = {
            'eldf': ELDF,
            'egdf': EGDF,
            'qldf': QLDF,
            'qgdf': QGDF,
        }
        cls = cls_map[self.gdf]
        kwargs = dict(
            DLB=self.DLB,
            DUB=self.DUB,
            LB=self.LB,
            UB=self.UB,
            S=self.S,
            z0_optimize=self.z0_optimize,
            tolerance=self.tolerance,
            data_form=self.data_form,
            n_points=self.n_points,
            homogeneous=self.homogeneous,
            catch=self.catch,
            weights=self.weights,
            wedf=self.wedf,
            opt_method=self.opt_method,
            verbose=self.verbose,
            max_data_size=self.max_data_size,
            flush=self.flush,
        )
        # Only local GDFs support variable-scale knobs.
        if self.gdf in ('eldf', 'qldf'):
            kwargs['varS'] = self.varS
            kwargs['minimum_varS'] = self.minimum_varS
        return cls(**kwargs)

    def initialize(self, x):
        # Prefer user/data-driven initialization source if provided.
        if self.data is not None:
            d = np.asarray(self.data).reshape(-1)
            if d.size == 0:
                base = np.ones(x.shape, dtype=np.float64)
            else:
                reps = int(np.ceil(x.size / d.size))
                base = np.tile(d, reps)[:x.size].reshape(x.shape).astype(np.float64)
        else:
            # fallback source when no data is given
            base = np.random.uniform(low=-1.0, high=1.0, size=x.shape)

        flat = base.reshape(-1)

        # Fit GDF on a bounded sample for speed/stability.
        if flat.size > self.max_data_size:
            idx = np.random.choice(flat.size, size=self.max_data_size, replace=False)
            z = flat[idx]
        else:
            z = flat

        try:
            gdf_model = self._build_gdf_model()
            gdf_model.fit(z)

            if hasattr(gdf_model, 'fj') and gdf_model.fj is not None:
                mod = np.asarray(gdf_model.fj, dtype=np.float64)
            elif hasattr(gdf_model, 'fi') and gdf_model.fi is not None:
                mod = np.asarray(gdf_model.fi, dtype=np.float64)
            else:
                mod = np.ones_like(z, dtype=np.float64)

            mod = mod.reshape(-1)
            if mod.size == 0:
                mod = np.ones_like(z, dtype=np.float64)
            if mod.size != flat.size:
                reps = int(np.ceil(flat.size / mod.size))
                mod = np.tile(mod, reps)[:flat.size]

            mod = np.abs(mod)
            s = np.sum(mod)
            if not np.isfinite(s) or s <= 0:
                mod = np.ones_like(mod)
            else:
                mod = mod / s * mod.size

            x[:] = (flat * mod).reshape(x.shape)
        except Exception:
            # safe fallback
            x[:] = base


class GDF(Gnostic):
    """Backward-compatible alias for `Gnostic` initializer."""
    pass


random_normal = RandomNormal()
random_uniform = RandomUniform()
zeros = Zeros()
ones = Ones()
truncated_normal = TruncatedNormal()