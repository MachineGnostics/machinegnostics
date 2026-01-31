'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2026  Nirmal Parmar

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar

Description:
This module implements Gnostic Local Clustering, a density-based clustering method 
using Estimating Local Distribution Functions (ELDF).
'''

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import logging

from machinegnostics.models.clustering.base_io_clustering import DataProcessClusteringBase
from machinegnostics.models.clustering.base_clustering_history import HistoryClusteringBase
from machinegnostics.magcal import ELDF
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.magcal import disable_parent_docstring

class GnosticLocalClustering(HistoryClusteringBase, DataProcessClusteringBase):
    """
    Gnostic Local Clustering using Estimating Local Distribution Functions (ELDF).

    This clustering model identifies clusters based on the local density of data points
    estimated by the ELDF. It detects modes (peaks) as cluster centers and valleys
    as cluster boundaries, naturally handling non-convex clusters and determining
    the number of clusters automatically.

    Key Features
    ------------
    - Density-based clustering: Uses ELDF to estimate the continuous probability density function.
    - Automatic cluster detection: Finds peaks (modes) and valleys (boundaries) to define clusters.
    - Grid search optimization: Can search for the optimal scale parameter (S) to minimize residual entropy.
    - Robustness: Inherits the robust properties of Gnostic Distribution Functions.
    - Interpretability: Provides clear visualization of PDF, peaks, and boundaries.

    Parameters
    ----------
    start_S : float, default=0.1
        The starting value for the scale parameter S in the grid search.
    end_S : float, default=2.0
        The ending value for the scale parameter S in the grid search.
    step_S : float, default=0.1
        The step size for the scale parameter S in the grid search.
    varS : bool, default=False
        If True, optimizes the minimum variance scale parameter instead of a fixed S.
    auto_S : bool, default=True
        If True, uses the automatic scale estimation capability of ELDF directly.
        If False, performs a grid search over the specified S range.
    verbose : bool, default=False
        If True, prints progress and diagnostics during fitting.
    history : bool, default=True
        If True, records the search history (e.g., S values, entropy, cluster counts).
    data_form : str, default='a'
        Internal data representation format ('a' for additive, 'm' for multiplicative).
        
    Attributes
    ----------
    centroids : np.ndarray
        Detected cluster modes (peaks of the PDF).
    labels : np.ndarray
        Cluster labels for each sample in the training data.
    cluster_boundaries : np.ndarray
        Detected boundaries between clusters (valleys of the PDF).
    optimal_S : float
        The scale parameter value of the selected best model.
    best_model : ELDF
        The fitted ELDF model instance corresponding to the best clustering.
    results : pd.DataFrame
        DataFrame containing the history of the grid search (if performed), including S values,
        number of clusters, and residual entropy.
    params : dict
        Dictionary containing the parameters and results of the best fitted model.

    Methods
    -------
    fit(X, y=None)
        Fit the Gnostic Local Clustering model to input features X.
    predict(X)
        Predict cluster labels for new input features X.
    score(X, y=None)
        Return the residual entropy of the fitted model.
    plot()
        Visualize the estimated PDF, detected clusters, and grid search results.

    Example
    -------
    >>> from machinegnostics.models import GnosticLocalClustering
    >>> model = GnosticLocalClustering(auto_S=True)
    >>> model.fit(X_train)
    >>> labels = model.predict(X_test)
    >>> model.plot()
    """

    @disable_parent_docstring
    def __init__(
        self,
        start_S: float = 0.1,
        end_S: float = 2.0,
        step_S: float = 0.1,
        varS: bool = False,
        auto_S: bool = True,
        verbose: bool = False,
        history: bool = True,
        data_form: str = 'a',
        **kwargs
    ):
        """
        Initialize the Gnostic Local Clustering model.
        
        Parameters
        ----------
        start_S : float, default=0.1
            Start of the S parameter search range.
        end_S : float, default=2.0
            End of the S parameter search range.
        step_S : float, default=0.1
            Step size for the S parameter search.
        varS : bool, default=False
            Whether to use variable scale parameter (heteroscedasticity).
        auto_S : bool, default=True
            Whether to use ELDF's auto-optimization for S.
        verbose : bool, default=False
            Enable verbose logging.
        history : bool, default=True
            Enable history recording.
        data_form : str, default='a'
            Data form: 'a' (additive) or 'm' (multiplicative).
        """
        super().__init__(
            verbose=verbose,
            history=history,
            data_form=data_form,
            **kwargs
        )
        
        self.start_S = start_S
        self.end_S = end_S
        self.step_S = step_S
        self.varS = varS
        self.auto_S = auto_S
        self.data_form = data_form # Explicitly store locally as well
        self.verbose = verbose
        self._record_history = history
        
        # Initialize attributes
        self.best_model = None
        self.optimal_S = None
        self.peaks_indices = None
        self.valleys_indices = None
        self.cluster_boundaries = None
        self.results = None
        self.centroids = None
        
        # Clear history initialized by parent and prepare for search history
        if self._record_history:
            self._history = []
        else:
            self._history = None

        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.info(f"{self.__class__.__name__} initialized.")

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit the Gnostic Local Clustering model to the provided data.

        Identifies clusters by fitting an ELDF to the data and finding peaks (centroids)
        and valleys (boundaries) in the estimated probability density function.

        Parameters
        ----------
        X : np.ndarray
            Input features. Must be 1D array for standard ELDF clustering.
        y : np.ndarray, optional
            Not used.

        Returns
        -------
        self : GnosticLocalClustering
            The fitted model.
        """
        self.logger.info("Starting fit process for GnosticLocalClustering.")
        
        # 1. IO check (Handling input format)
        Xc, _ = super()._fit_io(X, y)
        
        # Ensure 1D input for currently supported ELDF clustering
        if Xc.ndim > 1 and Xc.shape[1] > 1:
            # Flatten or handle multi-dim based on design choice. 
            # For now, ELDF is univariate. We warn if multi-dim.
            self.logger.warning("GnosticLocalClustering currently supports 1D data. Flattening input.")
            data = Xc.flatten()
        else:
            data = Xc.flatten()

        results_list = []

        # 2. Optimization Strategy (Auto vs Grid)
        if self.auto_S:
            self.logger.info("Running auto-S optimization.")
            try:
                # Use ELDF's internal auto-optimization
                # S='auto' is default, but explicit for clarity
                model = ELDF(S='auto', 
                             varS=self.varS, 
                             verbose=False, 
                             catch=True, 
                             data_form=self.data_form,
                             n_points=1000,
                             z0_optimize=False
                             )
                model.fit(data)
                
                s_val = model.params.get('S')
                if s_val is None:
                    s_val = model.S
                
                # Analyze PDF
                pdf = model.pdf_points
                peaks, _ = find_peaks(pdf)
                valleys, _ = find_peaks(-pdf)
                entropy = model.params.get('residual_entropy', np.nan)
                
                result_entry = {
                    'S': s_val,
                    'n_clusters': len(peaks),
                    'residual_entropy': entropy,
                    'model': model,
                    'peaks': peaks,
                    'valleys': valleys
                }
                results_list.append(result_entry)
                
                if self._record_history:
                    # Adapt to history format
                    self._history.append(result_entry)
                    
            except Exception as e:
                self.logger.error(f"Auto S fitting failed: {e}")
                raise e

        else:
            self.logger.info(f"Running grid search for S from {self.start_S} to {self.end_S}.")
            # Grid search logic
            s_values = np.arange(self.start_S, self.end_S + self.step_S/100, self.step_S)
            
            for i, s in enumerate(s_values):
                try:
                    # Configure ELDF
                    if self.varS:
                        model = ELDF(minimum_varS=float(s), 
                                     varS=True, 
                                     verbose=False, 
                                     catch=True, 
                                     data_form=self.data_form,
                                     n_points=1000,
                                     z0_optimize=False)
                    else:
                        model = ELDF(S=float(s), 
                                     varS=False, 
                                     verbose=False, 
                                     catch=True, 
                                     data_form=self.data_form,
                                     n_points=1000,
                                     z0_optimize=False)
                    
                    model.fit(data)
                    
                    # Analyze PDF
                    pdf = model.pdf_points
                    peaks, _ = find_peaks(pdf)
                    valleys, _ = find_peaks(-pdf)
                    
                    entropy = model.params.get('residual_entropy', np.nan)
                    
                    result_entry = {
                        'iteration': i,
                        'S': s,
                        'n_clusters': len(peaks),
                        'residual_entropy': entropy,
                        'model': model,
                        'peaks': peaks,
                        'valleys': valleys
                    }
                    results_list.append(result_entry)
                    
                    if self._record_history:
                        self._history.append(result_entry)
                        
                except Exception as e:
                    self.logger.warning(f"Skipping S={s:.2f} due to error: {e}")
                    continue

        # 3. Process Results
        self.results = pd.DataFrame(results_list)
        
        if self.results.empty:
            raise RuntimeError("No models converged during fitting.")
            
        # Select best model (Minimize Residual Entropy)
        best_idx = self.results['residual_entropy'].idxmin()
        best_row = self.results.loc[best_idx]
        
        self.optimal_S = best_row['S']
        self.best_model = best_row['model']
        self.peaks_indices = best_row['peaks']
        self.valleys_indices = best_row['valleys']
        
        # 4. Finalize Model Attributes
        # Cluster boundaries (valleys)
        domain = self.best_model.di_points_n
        self.cluster_boundaries = np.sort(domain[self.valleys_indices])
        
        # Centroids (peaks)
        self.centroids = domain[self.peaks_indices].reshape(-1, 1) # Reshape for consistency with 2D centroids
        self.n_clusters = len(self.centroids)
        
        # Labels for training data
        self.labels = self.predict(data)
        
        # Update params with best model parameters
        self.params = self.best_model.params.copy()
        self.params.update({
            'centroids': self.centroids,
            'cluster_boundaries': self.cluster_boundaries,
            'n_clusters': self.n_clusters,
            'S': self.optimal_S
        })
        
        self.logger.info(f"Fit complete. Optimal S: {self.optimal_S}, Clusters: {self.n_clusters}")
        return self

    def predict(self, model_input: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Assigns labels based on the intervals defined by the cluster boundaries (valleys).
        
        Parameters
        ----------
        model_input : np.ndarray
            Input features.

        Returns
        -------
        labels : np.ndarray
            Predicted cluster labels.
        """
        self.logger.info("Making predictions with GnosticLocalClustering.")
        
        if self.cluster_boundaries is None:
            raise ValueError("Model not fitted.")

        X_checked = super()._predict_io(model_input)
        
        # For now, flatten to 1D as we only support univariate clustering
        data = X_checked.flatten()
        
        # Use boundaries to bin data
        # Labels will be 0, 1, ..., n_boundaries mapping to clusters
        labels = np.searchsorted(self.cluster_boundaries, data)
        
        return labels

    def score(self, X: np.ndarray, y: np.ndarray = None) -> float:
        """
        Return the residual entropy of the best fitted model.
        
        Note: Unlike KMeans inertia (which is position-dependent), 
        ELDF minimizes residual entropy. Lower is better.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray, optional
            Ignored.

        Returns
        -------
        score : float
            Residual entropy of the model.
        """
        self.logger.info("Calculating score (entropy) with GnosticLocalClustering.")
        if self.best_model:
            return self.best_model.params.get('residual_entropy')
        return None

    def plot(self):
        """
        Plot the optimal ELDF, detected clusters, and grid search results.
        Display the PDF with marked peaks and valleys, and the Entropy/Cluster profile.
        """
        if self.best_model is None:
            print("Model not fitted.")
            return

        has_grid_data = len(self.results) > 1
        
        fig_size = (12, 10) if has_grid_data else (12, 6)
        fig, axes = plt.subplots(2 if has_grid_data else 1, 1, figsize=fig_size)
        
        if not has_grid_data:
             axes = [axes] # Make it subscriptable
        
        # --- Plot 1: PDF and Cluster Structure ---
        ax = axes[0]
        model = self.best_model
        # Plot PDF
        ax.plot(model.di_points_n, model.pdf_points, 'k-', lw=2, label='Estimated PDF')
        
        # Plot Peaks (Cluster Centers)
        px = model.di_points_n[self.peaks_indices]
        py = model.pdf_points[self.peaks_indices]
        ax.plot(px, py, 'ro', markersize=8, label='Peaks (Cluster Modes)')
        
        # Plot Valleys (Cluster Boundaries)
        vx = model.di_points_n[self.valleys_indices]
        vy = model.pdf_points[self.valleys_indices]
        ax.plot(vx, vy, 'bx', markersize=8, markeredgewidth=2, label='Valleys (Boundaries)')
        
        # Visualize Boundaries
        for b in self.cluster_boundaries:
            ax.axvline(b, color='blue', linestyle='--', alpha=0.4)
            
        optimal_s_val = self.optimal_S
        if hasattr(optimal_s_val, 'item'): 
             optimal_s_val = optimal_s_val.item()
        
        try:
            s_display = f"{float(optimal_s_val):.2f}"
        except (ValueError, TypeError):
            s_display = str(optimal_s_val)
             
        ax.set_title(f"Optimal Clustering (S={s_display}, Clusters={len(px)})")
        ax.set_xlabel("Data Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # --- Plot 2: Grid Search Analysis (Only if we did a grid search) ---
        if has_grid_data:
            ax2 = axes[1]
            ax3 = ax2.twinx()
            
            # Entropy
            l1 = ax2.plot(self.results['S'], self.results['residual_entropy'], 'g-o', label='Residual Entropy')
            # Cluster Count
            l2 = ax3.plot(self.results['S'], self.results['n_clusters'], 'm-s', label='Cluster Count')
            
            xlabel_text = 'Scale Parameter (S)' if not self.varS else 'Minimum VarS'
            ax2.set_xlabel(xlabel_text)
            ax2.set_ylabel('Residual Entropy', color='g')
            ax3.set_ylabel('Number of Clusters', color='m')
            
            # Legend merging
            lns = l1 + l2
            labs = [l.get_label() for l in lns]
            ax2.legend(lns, labs, loc='center right')
            
            ax2.set_title("Grid Search Profile: Entropy & Clusters")
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        """String representation of the GnosticLocalClustering instance."""
        return (f"GnosticLocalClustering(start_S={self.start_S}, end_S={self.end_S}, "
                f"step_S={self.step_S}, varS={self.varS}, auto_S={self.auto_S}, "
                f"verbose={self.verbose}, history={self._record_history})")
