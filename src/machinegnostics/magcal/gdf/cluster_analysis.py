'''
Cluster Analysis 

Module for clustering-based bound estimation for interval analysis.
This class do cluster end-to-end cluster analysis to estimate bounds.

Authors: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
import warnings
from machinegnostics.magcal import ELDF, EGDF, DataCluster, DataHomogeneity

class ClusterAnalysis:
    '''
    Cluster Analysis for bound estimation.

    Parameters
    ----------
    gdf : ELDF
        An instance of the ELDF class containing the data.
    verbose : bool, optional
        If True, prints detailed logs during processing. Default is False.
    catch : bool, optional
        If True, stores intermediate results for further analysis. Default is False.
    cluster_bounds : bool, optional
        If True, performs clustering to estimate bounds. Default is True.

    Attributes
    ----------
    LCB : float or None
        Lower cluster bound estimated from the data.
    UCB : float or None
        Upper cluster bound estimated from the data.
    '''

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

    def _add_warning(self, warning: str):
        self.params['warnings'].append(warning)
        if self.verbose:
            print(f'ClusterAnalysis: Warning: {warning}')
    
    def _add_error(self, error: str):
        self.params['error'].append(error)
        if self.verbose:
            print(f'ClusterAnalysis: Error: {error}')

    def fit(self, data: np.ndarray, plot: bool = False) -> tuple:
        '''
        Fit the Cluster Analysis model to estimate bounds.

        Returns
        -------
        tuple
            A tuple containing the lower and upper cluster bounds (LCB, UCB).
        '''
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
            if self.verbose:
                print("ClusterAnalysis: Fitting EGDF...")
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
            if self.verbose:
                print("ClusterAnalysis: Fitting ELDF...")
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
            if self.verbose:
                print("ClusterAnalysis: Estimating cluster bounds...")

            # note for user, if is_homogeneous is False, LCB and UCB will provide main cluster of the data.
            if not is_homogeneous:
                info_msg = "Data is not homogeneous, LCB and UCB will provide bounds for the main cluster of the data."
                self._add_warning(info_msg)
                if self.verbose:
                    print(f'ClusterAnalysis: Info: {info_msg}')

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
                if self.verbose:
                    print("ClusterAnalysis: Data flushed to save memory.")

            if self.verbose:
                print(f'ClusterAnalysis: Fitting completed. LCB: {self.LCB}, UCB: {self.UCB}')
            return self.LCB, self.UCB
        
        except Exception as e:
            self._add_error(str(e))
            if self.verbose:
                print(f'ClusterAnalysis: Error during fit: {e}')
            return None, None

    def results(self) -> dict:
        '''
        Get the results of the Cluster Analysis.

        Returns
        -------
        dict
            A dictionary containing the estimated bounds.
        '''
        if not self._fitted:
            raise RuntimeError("ClusterAnalysis: The model is not fitted yet. Please call the 'fit' method first.")
        return {
            'LCB': float(self.LCB),
            'UCB': float(self.UCB)
        }
    
    def plot(self) -> None:
        '''
        Plot the ELDF and DataCluster results.

        Returns
        -------
        None
        '''
        if not self._fitted:
            raise RuntimeError("ClusterAnalysis: The model is not fitted yet. Please call the 'fit' method first.")
        
        # if flush is True, raise error
        if self.flush:
            raise RuntimeError("ClusterAnalysis: Data has been flushed. Cannot plot. Please set 'flush' to False during initialization to enable plotting.")

        # Plot ELDF
        self._eldf.plot(plot='both')

        # Plot DataCluster
        self._data_cluster.plot()