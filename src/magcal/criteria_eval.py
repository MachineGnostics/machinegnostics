import numpy as np
from src.magcal.characteristics import GnosticsCharacteristics
from src.magcal.sample_characteristics import GnosticCharacteristicsSample

class CriteriaEvaluator:
    """
    A class to evaluate the performance of a model's fit to data using various statistical and information-theoretic metrics.

    This class computes several evaluation metrics, including:
    - Robust R-squared (RobR2): A robust measure of the goodness of fit.
    - Geometric Mean of Model Fit Error (GMMFE): A measure of the average relative error between the observed and fitted values.
    - Divergence Information (DivI): A measure of the divergence between the distributions of observed and fitted values.
    - Evaluation Metric (EvalMet): A composite metric combining RobR2, GMMFE, and DivI.

    The class also provides a method to generate a report summarizing these metrics.

    Attributes:
        y (np.ndarray): The observed data (ground truth).
        y_fit (np.ndarray): The fitted data (model predictions).
        w (np.ndarray): Weights for the data points. Defaults to an array of ones if not provided.
        robr2 (float): The computed Robust R-squared value. Initialized to None.
        gmmfe (float): The computed Geometric Mean of Model Fit Error. Initialized to None.
        divI (float): The computed Divergence Information value. Initialized to None.
        evalmet (float): The computed Evaluation Metric. Initialized to None.
        _report (dict): A dictionary containing the computed metrics. Initialized to an empty dictionary.

    Methods:
        __init__(y, y_fit, w=None):
            Initializes the CriteriaEvaluator with observed data, fitted data, and optional weights.

        _robr2():
            Computes the Robust R-squared (RobR2) value. This metric measures the proportion of variance in the observed data
            explained by the fitted data, with robustness to outliers.

        _gmmfe():
            Computes the Geometric Mean of Model Fit Error (GMMFE). This metric quantifies the average relative error between
            the observed and fitted values on a logarithmic scale.

        _divI():
            Computes the Divergence Information (DivI). This metric measures the divergence between the distributions of the
            observed and fitted values using gnostic characteristics.

        _evalmet():
            Computes the Evaluation Metric (EvalMet) as a composite measure combining RobR2, GMMFE, and DivI.

        _generate_report():
            Generates a report summarizing all computed metrics (RobR2, GMMFE, DivI, and EvalMet) in a dictionary format.

    Usage:
        evaluator = CriteriaEvaluator(y, y_fit, w)
        robr2 = evaluator._robr2()
        gmmfe = evaluator._gmmfe()
        divI = evaluator._divI()
        evalmet = evaluator._evalmet()
        report = evaluator._generate_report()

    Notes:
        - The class assumes that `y` and `y_fit` are non-negative and of the same shape.
        - The methods `_robr2`, `_gmmfe`, `_divI`, and `_evalmet` are designed to be called internally, but they can be
          invoked directly if needed.
        - The `_generate_report` method ensures that all metrics are computed before generating the report.
    """
    def __init__(self, y, y_fit, w=None):
        self.y = np.asarray(y)
        self.y_fit = np.asarray(y_fit)
        self.w = np.ones_like(self.y) if w is None else np.asarray(w)
        self.robr2 = None
        self.gmmfe = None
        self.divI = None
        self.evalmet = None
        self._report = {}

    def _robr2(self):
        e = self.y - self.y_fit
        e_bar = np.sum(self.w * e) / np.sum(self.w)
        y_bar = np.sum(self.w * self.y) / np.sum(self.w)
        num = np.sum(self.w * (e - e_bar) ** 2)
        denom = np.sum(self.w * (self.y - y_bar) ** 2)
        self.robr2 = 1 - num / denom if denom != 0 else 0.0
        return self.robr2

    def _gmmfe(self):
        epsilon = 1e-10  # Small value to prevent division by zero
        log_ratios = np.abs(np.log(self.y / (self.y_fit + epsilon)))
        self.gmmfe = np.exp(np.mean(log_ratios))
        return self.gmmfe

    def _divI(self):
        gcs_y = GnosticCharacteristicsSample(data=self.y)
        gcs_y_fit = GnosticCharacteristicsSample(data=self.y_fit)

        y_median = gcs_y._gnostic_median(case='i').root
        y_fit_median = gcs_y_fit._gnostic_median(case='i').root

        zy = self.y / y_median
        zf = self.y_fit / y_fit_median

        gc_y = GnosticsCharacteristics(zy)
        gc_y_fit = GnosticsCharacteristics(zf)

        qy, q1y = gc_y._get_q_q1()
        qf, q1f = gc_y_fit._get_q_q1()

        hi = gc_y._hi(q=qy, q1=q1y)
        hi_fit = gc_y_fit._hi(q=qf, q1=q1f)

        pi = gc_y._idistfun(hi)
        pi_fit = gc_y_fit._idistfun(hi_fit)

        epsilon = 1e-10  # Small value to prevent log(0)
        pi = np.clip(pi, epsilon, 1 - epsilon)  # Clip values to avoid invalid log
        pi_fit = np.clip(pi_fit, epsilon, 1 - epsilon)  # Clip values to avoid invalid log

        Iy = gc_y._info_i(pi)
        Iy_fit = gc_y_fit._info_i(pi_fit)

        self.divI = np.mean(Iy / Iy_fit)
        return self.divI

    def _evalmet(self):
        if self.robr2 is None:
            self._robr2()
        if self.gmmfe is None:
            self._gmmfe()
        if self.divI is None:
            self._divI()
        self.evalmet = self.robr2 / (self.gmmfe * self.divI)
        return self.evalmet

    def _generate_report(self):
        if self.robr2 is None:
            self._robr2()
        if self.gmmfe is None:
            self._gmmfe()
        if self.divI is None:
            self._divI()
        if self.evalmet is None:
            self._evalmet()

        self._report = {
            "RobR2": self.robr2,
            "GMMFE": self.gmmfe,
            "DivI": self.divI,
            "EvalMet": self.evalmet
        }
        return self._report