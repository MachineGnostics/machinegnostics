'''
Cluster Analysis 

Module for clustering-based bound estimation for interval analysis.
This class do cluster end-to-end cluster analysis to estimate bounds.

Authors: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
import warnings
import logging
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.magcal import ELDF, EGDF, DataCluster, DataHomogeneity

class ClusterAnalysis:
    """
    ClusterAnalysis - End-to-End Clustering-Based Bound Estimation for Interval Analysis

    The `ClusterAnalysis` class provides a high-level, automated workflow for estimating the main cluster bounds of a dataset using Gnostic Distribution Functions (GDFs) and advanced clustering analysis. It is designed for robust, interpretable, and reproducible interval estimation in scientific, engineering, and data science applications.

    This class orchestrates the entire process of:
      1. Fitting a GDF (typically ELDF/EGDF) to the data,
      2. Assessing data homogeneity,
      3. Performing cluster boundary detection using the DataCluster algorithm,
      4. Returning interpretable lower and upper cluster bounds (LCB, UCB) for the main data cluster.

    **Key Features:**
      - Fully automated pipeline for cluster-based bound estimation
      - Integrates GDF fitting, homogeneity testing, and cluster analysis
      - Supports both local (ELDF) and global (EGDF) GDFs
      - Handles weighted data, bounded/unbounded domains, and advanced parameterization
      - Detailed error/warning logging and reproducible parameter tracking
      - Optional memory-efficient operation via flushing intermediate results
      - Visualization support for both GDF and cluster analysis results

    Parameters
    ----------
    verbose : bool, optional (default=False)
        If True, prints detailed logs and progress information during processing.
    catch : bool, optional (default=True)
        If True, stores intermediate results and diagnostic information for further analysis.
    derivative_threshold : float, optional (default=0.01)
        Threshold for derivative-based cluster boundary detection (used by DataCluster).
    slope_percentile : int, optional (default=70)
        Percentile threshold for slope-based boundary detection (used by DataCluster).
    DLB : float, optional
        Data Lower Bound. If None, inferred from data.
    DUB : float, optional
        Data Upper Bound. If None, inferred from data.
    LB : float, optional
        Lower probable bound for the distribution.
    UB : float, optional
        Upper probable bound for the distribution.
    S : float or str, optional (default='auto')
        Scale parameter for the GDF. Use 'auto' for automatic estimation.
    varS : bool, optional (default=False)
        If True, allows variable scale parameter during optimization.
    z0_optimize : bool, optional (default=True)
        If True, optimizes the location parameter Z0 during fitting.
    tolerance : float, optional (default=1e-5)
        Convergence tolerance for optimization.
    data_form : str, optional (default='a')
        Data processing form: 'a' for additive, 'm' for multiplicative.
    n_points : int, optional (default=1000)
        Number of points for GDF evaluation and PDF generation.
    homogeneous : bool, optional (default=True)
        If True, assumes data is homogeneous; triggers warnings if not.
    weights : np.ndarray, optional
        Prior weights for data points. If None, uniform weights are used.
    wedf : bool, optional (default=False)
        If True, uses Weighted Empirical Distribution Function.
    opt_method : str, optional (default='L-BFGS-B')
        Optimization method for parameter estimation.
    max_data_size : int, optional (default=1000)
        Maximum data size for smooth GDF generation.
    flush : bool, optional (default=False)
        If True, flushes intermediate results after fitting to save memory.

    Attributes
    ----------
    LCB : float or None
        Lower Cluster Bound estimated from the data (main cluster lower edge).
    UCB : float or None
        Upper Cluster Bound estimated from the data (main cluster upper edge).
    params : dict
        Dictionary containing all parameters, intermediate results, errors, and warnings.
    _fitted : bool
        Indicates whether the analysis has been successfully completed.

    Methods
    -------
    fit(data: np.ndarray, plot: bool = False) -> tuple
        Runs the full cluster analysis pipeline on the input data.
        Returns (LCB, UCB) as the main cluster bounds.
    results() -> dict
        Returns a dictionary with the estimated bounds and key results.
    plot() -> None
        Visualizes the fitted GDF and cluster analysis results (if not flushed).

    Workflow
    --------
    1. Initialize ClusterAnalysis with desired parameters (no data required).
    2. Call `fit(data)` to perform the complete analysis and estimate cluster bounds.
    3. Access results via `results()` or visualize with `plot()`.

    Example
    -------
    >>> from machinegnostics.magcal import ClusterAnalysis
    >>> data = np.random.normal(0, 1, 1000)
    >>> ca = ClusterAnalysis(verbose=True)
    >>> LCB, UCB = ca.fit(data)
    >>> print(f"Main cluster bounds: LCB={LCB:.3f}, UCB={UCB:.3f}")
    >>> ca.plot()
    >>> results = ca.results()
    >>> print(results)

    Notes
    -----
    - The class is designed for robust, interpretable cluster-based bound estimation.
    - Works best with local GDFs (ELDF); global GDFs (EGDF) are supported but less flexible for clustering.
    - If `homogeneous=True` but the data is found to be heterogeneous, a warning is issued.
    - All intermediate parameters, errors, and warnings are tracked in `params` for reproducibility.
    - For large datasets or memory-constrained environments, set `flush=True` to save memory (but disables plotting).

    Raises
    ------
    RuntimeError
        If results or plot are requested before fitting.
    Exception
        If any step in the pipeline fails (errors are logged in `params`).

    References
    ----------
    - Gnostic Distribution Function theory and clustering methods (see mathematical gnostics literature).
    - For details on the underlying algorithms, see the documentation for ELDF, EGDF, and DataCluster classes.
    """

    def __init__(self,
                verbose: bool = False,
                catch: bool = True,
                derivative_threshold: float = 0.01,
                slope_percentile: int = 70,
                DLB: float = None,
                DUB: float = None,
                LB: float = None,
                UB: float = None,
                S: str = 'auto',
                varS: bool = False,
                z0_optimize: bool = True,
                tolerance: float = 0.00001,
                data_form: str = 'a',
                n_points: int = 1000,
                homogeneous: bool = True,
                weights: np.ndarray = None,
                wedf: bool = False,
                opt_method: str = 'L-BFGS-B',
                max_data_size: int = 1000,
                flush: bool = False
                ):
        ELDF.__init__(self)
        self.verbose = verbose
        self.catch = catch
        self.derivative_threshold = derivative_threshold
        self.slope_percentile = slope_percentile
        self.DLB = DLB
        self.DUB = DUB
        self.LB = LB
        self.UB = UB
        self.S = S
        self.varS = varS
        self.z0_optimize = z0_optimize
        self.tolerance = tolerance
        self.data_form = data_form
        self.n_points = n_points
        self.homogeneous = homogeneous
        self.weights = weights
        self.wedf = wedf
        self.opt_method = opt_method
        self.max_data_size = max_data_size
        self.flush = flush

        self._fitted = False

        self.LCB = None
        self.UCB = None

        self.params = {}
        self.params['error'] = []
        self.params['warnings'] = []

        # append arguments to params
        if self.catch:
            self.params['ClusterAnalysis'] = {
                'verbose': self.verbose,
                'catch': self.catch,
                'derivative_threshold': self.derivative_threshold,
                'slope_percentile': self.slope_percentile,
                'DLB': self.DLB,
                'DUB': self.DUB,
                'LB': self.LB,
                'UB': self.UB,
                'S': self.S,
                'varS': self.varS,
                'z0_optimize': self.z0_optimize,
                'tolerance': self.tolerance,
                'data_form': self.data_form,
                'n_points': self.n_points,
                'homogeneous': self.homogeneous,
                'weights': self.weights,
                'wedf': self.wedf,
                'opt_method': self.opt_method,
                'max_data_size': self.max_data_size,
                'flush': self.flush
            }
        
        # logger setup
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")

    def _add_warning(self, warning: str):
        self.params['warnings'].append(warning)
        self.logger.warning(warning)
    
    def _add_error(self, error: str):
        self.params['error'].append(error)
        self.logger.error(error)

    def fit(self, data: np.ndarray, plot: bool = False) -> tuple:
        """
        Fit the ClusterAnalysis model to the input data and estimate main cluster bounds.

        This method performs a comprehensive, automated clustering-based interval analysis on the provided data.
        It integrates Gnostic Distribution Function (GDF) fitting, homogeneity assessment, and advanced cluster boundary detection
        to robustly estimate the lower and upper bounds (LCB, UCB) of the main data cluster.

        The process is designed to be robust, interpretable, and reproducible, handling both homogeneous and heterogeneous data,
        supporting weighted and bounded datasets, and providing detailed diagnostics and error handling.
        All intermediate results, warnings, and errors are tracked in the `params` attribute for transparency.

        Parameters
        ----------
        data : np.ndarray
            Input data array for interval analysis. Should be 1D and numeric.
        plot : bool, optional (default=False)
            If True, generates plots for the fitted GDF and cluster analysis.

        Returns
        -------
        tuple
            (LCB, UCB): The estimated lower and upper bounds of the main data cluster.
            Returns (None, None) if fitting fails.

        Notes
        -----
        - If the data is found to be heterogeneous but `homogeneous=True`, a warning is issued.
        - If `flush=True`, intermediate objects are deleted after fitting to save memory (disables plotting).
        - All errors and warnings are logged in `params` for reproducibility and debugging.

        Raises
        ------
        Exception
            Any error during the fitting process is caught, logged, and (LCB, UCB) is set to (None, None).

        Example
        -------
        >>> ca = ClusterAnalysis(verbose=True)
        >>> LCB, UCB = ca.fit(data)
        >>> print(f"Cluster bounds: LCB={LCB}, UCB={UCB}")
        """
        self.logger.info("Starting ClusterAnalysis fit process...")
        try:
            kwrgs_egdf = {
                "DLB": self.DLB,
                "DUB": self.DUB,
                "LB": self.LB,
                "UB": self.UB,
                "S": self.S,
                "z0_optimize": self.z0_optimize,
                "tolerance": self.tolerance,
                "data_form": self.data_form,
                "n_points": self.n_points,
                "homogeneous": self.homogeneous,
                "catch": self.catch,
                "weights": self.weights,
                "wedf": self.wedf,
                "opt_method": self.opt_method,
                "verbose": self.verbose,
                "max_data_size": self.max_data_size,
                "flush": self.flush
                }
            # estimate egdf
            self.logger.info("ClusterAnalysis: Fitting EGDF...")
            self._egdf = EGDF(**kwrgs_egdf)
            self._egdf.fit(data, plot=False)
            if self.catch:
                self.params['EGDF'] = self._egdf.params

            # check data homogeneity
            self._data_homogeneity = DataHomogeneity(gdf=self._egdf,
                                                    verbose=self.verbose,
                                                    catch=self.catch,
                                                    flush=self.flush)
            is_homogeneous = self._data_homogeneity.fit(plot=False)
            if self.catch:
                self.params['DataHomogeneity'] = self._data_homogeneity.params

            # if self.homogeneous is True, and is_homogeneous is False, raise a warning for user, that user understanding for data may not be correct
            if self.homogeneous and not is_homogeneous:
                warning_msg = "Data is not homogeneous, but 'homogeneous' parameter is set to True. User understanding for data may not be correct."
                self._add_warning(warning_msg)
                warnings.warn(warning_msg)
        
            # fit eldf
            self.logger.info("ClusterAnalysis: Fitting ELDF...")
            self._eldf = ELDF(DLB=self.DLB,
                            DUB=self.DUB,
                            LB=self.LB,
                            UB=self.UB,
                            S=self.S,
                            varS=self.varS,
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
                            flush=self.flush)
            self._eldf.fit(data, plot=False)
            if self.catch:
                self.params['ELDF'] = self._eldf.params

            # get cluster bounds
            self.logger.info("ClusterAnalysis: Estimating cluster bounds...")

            # note for user, if is_homogeneous is False, LCB and UCB will provide main cluster of the data.
            if not is_homogeneous:
                info_msg = "Data is not homogeneous, LCB and UCB will provide bounds for the main cluster of the data."
                self._add_warning(info_msg)
                self.logger.info(f'ClusterAnalysis: Info: {info_msg}')

            self._data_cluster = DataCluster(gdf=self._eldf, 
                                            verbose=self.verbose, 
                                            catch=self.catch, 
                                            derivative_threshold=self.derivative_threshold, slope_percentile=self.slope_percentile)
            self.LCB, self.UCB = self._data_cluster.fit(plot=plot)
            if self.catch:
                self.params['DataCluster'] = self._data_cluster.params

            # save results
            self._fitted = True
            if self.catch:
                self.params['results'] = {
                    'LCB': self.LCB,
                    'UCB': self.UCB
                }

            # flush
            if self.flush:
                self._egdf = None
                self._eldf = None
                self._data_homogeneity = None
                self._data_cluster = None
                # deleter respective params to save memory
                # keep erros and warnings
                if self.catch:
                    if 'EGDF' in self.params:
                        del self.params['EGDF']
                    if 'ELDF' in self.params:
                        del self.params['ELDF']
                    if 'DataHomogeneity' in self.params:
                        del self.params['DataHomogeneity']
                    if 'DataCluster' in self.params:
                        del self.params['DataCluster']
                self.logger.info("ClusterAnalysis: Data flushed to save memory.")

            self.logger.info(f'ClusterAnalysis: Fitting completed. LCB: {self.LCB}, UCB: {self.UCB}')
            return self.LCB, self.UCB
        
        except Exception as e:
            self._add_error(str(e))
            self.logger.error(f'ClusterAnalysis: Error during fit: {e}')
            return None, None

    def results(self) -> dict:
        """
        Get the results of the Cluster Analysis.

        Returns
        -------
        dict
            A dictionary containing the estimated lower and upper cluster bounds:
            {
                'LCB': float,
                'UCB': float
            }

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.

        Notes
        -----
        - Call this method after `fit()` to retrieve the main cluster bounds.
        - The returned values are floats or None if fitting failed.
        """
        self.logger.info("Retrieving ClusterAnalysis results...")
        if not self._fitted:
            self.logger.error("ClusterAnalysis: The model is not fitted yet. Please call the 'fit' method first.")
            raise RuntimeError("ClusterAnalysis: The model is not fitted yet. Please call the 'fit' method first.")
        return {
            'LCB': float(self.LCB),
            'UCB': float(self.UCB)
        }

    def plot(self) -> None:
        """
        Plot the ELDF and DataCluster results.

        This method visualizes the fitted Empirical Local Distribution Function (ELDF)
        and the detected cluster boundaries from DataCluster. It is only available if
        the model has been fitted and intermediate data has not been flushed.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet, or if `flush=True` was set and intermediate data is unavailable.

        Notes
        -----
        - Call this method after `fit()` to visualize the results.
        - If `flush=True` was set during initialization, plotting is disabled to save memory.
        """
        if not self._fitted:
            self.logger.error("ClusterAnalysis: The model is not fitted yet. Please call the 'fit' method first.")
            raise RuntimeError("ClusterAnalysis: The model is not fitted yet. Please call the 'fit' method first.")
        if self.flush:
            self.logger.error("ClusterAnalysis: Data has been flushed. Cannot plot. Please set 'flush' to False during initialization to enable plotting.")
            raise RuntimeError("ClusterAnalysis: Data has been flushed. Cannot plot. Please set 'flush' to False during initialization to enable plotting.")

        # Plot ELDF
        self._eldf.plot(plot='both')

        # Plot DataCluster
        self._data_cluster.plot()