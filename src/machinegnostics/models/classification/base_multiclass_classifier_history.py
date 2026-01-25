'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
Date: 2025-01-25

Description:
History class for the Multiclass Classifier model.

This class extends MulticlassClassifierCalBase to maintain a history
of model parameters and gnostic loss values during training iterations.
'''

import numpy as np
from machinegnostics.models.classification.base_multiclass_classifier_cal import MulticlassClassifierCalBase

class HistoryMulticlassClassifierBase(MulticlassClassifierCalBase):
    """
    History class for the Multiclass Classifier model.
    
    This class extends MulticlassClassifierCalBase to maintain a history
    of model parameters and gnostic loss values during training iterations.
    
    Parameters needed to record history:
        - loss: Gnostic loss value at each iteration
        - iteration: The iteration number
        - weights: Model weights at each iteration
        - coefficients: Model coefficients at each iteration
        - degree: Degree of polynomial features used in the model
        - rentropy: Entropy of the model at each iteration
        - cross_entropy_loss: Cross-entropy loss at each iteration
        - fi, hi, fj, hj, infoi, infoj, pi, pj, ei, ej: Additional gnostic information if calculated
    """
    
    def __init__(self,
                 degree: int = 1,
                 max_iter: int = 100,
                 tolerance: float = 1e-1,
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: 'str | int | float' = 'auto',
                 data_form: str = 'a',
                 gnostic_characteristics: bool = True,
                 history: bool = True):
        super().__init__(
            degree=degree,
            max_iter=max_iter,
            tolerance=tolerance,
            early_stopping=early_stopping,
            verbose=verbose,
            scale=scale,
            data_form=data_form,
            gnostic_characteristics=gnostic_characteristics,
            history=history
        )
        
        self.degree = degree
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.scale = scale
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self.history = history
        
        # Initialize params for history tracking
        self.params = [
            {
                'iteration': 0,
                'loss': None,
                'weights': None,
                'coefficients': None,
                'degree': self.degree,
                'rentropy': None,
                'cross_entropy_loss': None,
                'fi': None,
                'hi': None,
                'fj': None,
                'hj': None,
                'infoi': None,
                'infoj': None,
                'pi': None,
                'pj': None,
                'ei': None,
                'ej': None
            }
        ]

        # logger
        self.logger.info(f"{self.__class__.__name__} initialized.")
    
    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the data and record history.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        """
        self.logger.info(f"Starting fit process for {self.__class__.__name__}.")
        # Call the parent fit method to perform fitting
        super()._fit(X, y)

        # Record the final state in history as a dict
        params_dict = {}

        if self.gnostic_characteristics:
            params_dict['iteration'] = self._iter + 1
            params_dict['loss'] = self.loss
            params_dict['weights'] = self.weights.copy() if self.weights is not None else None
            params_dict['coefficients'] = self.coefficients.copy() if self.coefficients is not None else None
            params_dict['degree'] = self.degree
            params_dict['rentropy'] = self.re
            params_dict['cross_entropy_loss'] = self.cross_entropy_loss
            params_dict['fi'] = self.fi
            params_dict['hi'] = self.hi
            params_dict['fj'] = self.fj
            params_dict['hj'] = self.hj
            params_dict['infoi'] = self.infoi
            params_dict['infoj'] = self.infoj
            params_dict['pi'] = self.pi
            params_dict['pj'] = self.pj
            params_dict['ei'] = self.ei
            params_dict['ej'] = self.ej
        else:
            params_dict['iteration'] = self._iter + 1
            params_dict['loss'] = None
            params_dict['weights'] = self.weights.copy() if self.weights is not None else None
            params_dict['coefficients'] = self.coefficients.copy() if self.coefficients is not None else None
            params_dict['degree'] = self.degree
            params_dict['rentropy'] = self.re if hasattr(self, 're') else None
            params_dict['cross_entropy_loss'] = self.cross_entropy_loss if hasattr(self, 'cross_entropy_loss') else None

        self.params.append(params_dict)
