'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.sample_characteristics import GnosticCharacteristicsSample
import logging 
from machinegnostics.magcal.util.logging import get_logger

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

        generate_report():
            Generates a report summarizing all computed metrics (RobR2, GMMFE, DivI, and EvalMet) in a dictionary format.

    Usage:
        Example 1: Basic Usage
        -----------------------
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_fit = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        evaluator = CriteriaEvaluator(y, y_fit)

        robr2 = evaluator._robr2()
        print("Robust R-squared:", robr2)

        gmmfe = evaluator._gmmfe()
        print("Geometric Mean of Model Fit Error:", gmmfe)

        divI = evaluator._divI()
        print("Divergence Information:", divI)

        evalmet = evaluator._evalmet()
        print("Evaluation Metric:", evalmet)

        report = evaluator.generate_report()
        print("Report:", report)

        Example 2: Using Weights
        ------------------------
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_fit = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        weights = np.array([1, 2, 1, 2, 1])
        evaluator = CriteriaEvaluator(y, y_fit, w=weights)

        robr2 = evaluator._robr2()
        print("Weighted Robust R-squared:", robr2)

        gmmfe = evaluator._gmmfe()
        print("Weighted Geometric Mean of Model Fit Error:", gmmfe)

        divI = evaluator._divI()
        print("Weighted Divergence Information:", divI)

        evalmet = evaluator._evalmet()
        print("Weighted Evaluation Metric:", evalmet)

        report = evaluator.generate_report()
        print("Weighted Report:", report)

    Notes:
        - The class assumes that `y` and `y_fit` are non-negative and of the same shape.
        - The methods `_robr2`, `_gmmfe`, `_divI`, and `_evalmet` are designed to be called internally, but they can be
          invoked directly if needed.
        - The `generate_report` method ensures that all metrics are computed before generating the report.
    """
    def __init__(self, y, y_fit, w=None, verbose: bool = False):
        self.y = np.asarray(y)
        self.y_fit = np.asarray(y_fit)
        self.w = np.ones_like(self.y) if w is None else np.asarray(w)
        self.robr2 = None
        self.gmmfe = None
        self.divI = None
        self.evalmet = None
        self._report = {}

        # logger setup
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")

    def _robr2(self):
        """
        Computes the Robust R-squared (RobR2) value.

        This metric measures the proportion of variance in the observed data explained by the fitted data,
        with robustness to outliers. It is calculated as:

        RobR2 = 1 - (Weighted Sum of Squared Errors / Weighted Total Sum of Squares)

        Returns:
            float: The computed Robust R-squared value. A value closer to 1 indicates a better fit.
        """
        self.logger.info("Calculating Robust R-squared (RobR2).")
        e = self.y - self.y_fit
        e_bar = np.sum(self.w * e) / np.sum(self.w)
        y_bar = np.sum(self.w * self.y) / np.sum(self.w)
        num = np.sum(self.w * (e - e_bar) ** 2)
        denom = np.sum(self.w * (self.y - y_bar) ** 2)
        self.robr2 = 1 - num / denom if denom != 0 else 0.0
        self.logger.info(f"Robust R-squared (RobR2) calculated: {self.robr2}")
        return self.robr2

    def _gmmfe(self):
        """
        Computes the Geometric Mean of Model Fit Error (GMMFE).

        This metric quantifies the average relative error between the observed and fitted values on a logarithmic scale.
        It is robust to outliers and provides a measure of the average multiplicative error.

        Returns:
            float: The computed GMMFE value. A value closer to 1 indicates a better fit.
        """
        self.logger.info("Calculating Geometric Mean of Model Fit Error (GMMFE).")
        epsilon = 1e-10  # Small value to prevent division by zero
        # avoid log failure
        zz = self.y / (self.y_fit + epsilon)
        zz = np.clip(zz, epsilon, None)  # Clip values to avoid invalid log
        log_ratios = np.abs(np.log(zz))
        # avoid exp failure
        log_ratios = np.clip(log_ratios, None, 100)  # Clip values to avoid invalid exp
        self.gmmfe = np.exp(np.mean(log_ratios))
        self.logger.info(f"Geometric Mean of Model Fit Error (GMMFE) calculated: {self.gmmfe}")
        return self.gmmfe

    def _divI(self):
        """
        Computes the Divergence Information (DivI).

        This metric measures the divergence between the distributions of the observed and fitted values using gnostic characteristics.
        It involves calculating the gnostic characteristics of the observed and fitted data, and then computing the divergence
        in their information content.

        Returns:
            float: The computed Divergence Information value. A lower value indicates less divergence and a better fit.
        """
        self.logger.info("Calculating Divergence Information (DivI).")
        gcs_y = GnosticCharacteristicsSample(data=self.y)
        gcs_y_fit = GnosticCharacteristicsSample(data=self.y_fit)

        # y_median = gcs_y._gnostic_median(case='i').root
        # y_fit_median = gcs_y_fit._gnostic_median(case='i').root

        y_median = np.median(self.y) # Using numpy median for simplicity NOTE
        y_fit_median = np.median(self.y_fit)

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
        self.logger.info(f"Divergence Information (DivI) calculated: {self.divI}")
        return self.divI

    def _evalmet(self):
        """
        Computes the Evaluation Metric (EvalMet).

        This is a composite metric that combines Robust R-squared (RobR2), Geometric Mean of Model Fit Error (GMMFE),
        and Divergence Information (DivI). It is calculated as:

        EvalMet = RobR2 / (GMMFE * DivI)

        Returns:
            float: The computed Evaluation Metric. A higher value indicates a better overall fit.
        """
        self.logger.info("Calculating Evaluation Metric (EvalMet).")
        if self.robr2 is None:
            self._robr2()
        if self.gmmfe is None:
            self._gmmfe()
        if self.divI is None:
            self._divI()
        self.evalmet = self.robr2 / (self.gmmfe * self.divI)
        self.logger.info(f"Evaluation Metric (EvalMet) calculated: {self.evalmet}")
        return self.evalmet

    def generate_report(self):
        """
        Generates a report summarizing all computed metrics.

        This method ensures that all metrics (RobR2, GMMFE, DivI, and EvalMet) are computed before generating the report.
        The report is returned as a dictionary containing the metric names and their corresponding values.

        Returns:
            dict: A dictionary containing the computed metrics:
                - "RobR2": Robust R-squared value.
                - "GMMFE": Geometric Mean of Model Fit Error.
                - "DivI": Divergence Information.
                - "EvalMet": Evaluation Metric.
        """
        self.logger.info("Generating evaluation report.")
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
        self.logger.info("Evaluation report generated.")
        return self._report