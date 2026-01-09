'''
Gnostic - Homoscedasticity and Heteroscedasticity

This module to check for homoscedasticity and heteroscedasticity in data.

Primary work with ELDF and QLDF classes (local gnostics distribution functions).
Fit GDF for given data with varS=True to estimate variable Scale parameter.
If Scale parameter is variable, data is heteroscedastic. If Scale parameter is constant, data is homoscedastic.

Author: Nirmal Parmar
Machine Gnostics
'''
import numpy as np
import logging
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.magcal import ELDF, QLDF
from typing import Union
class DataScedasticity:
    """
    Gnostic scedasticity test for homoscedasticity vs. heteroscedasticity.

    This class determines whether a dataset is homoscedastic (constant scale)
    or heteroscedastic (varying scale) using a fitted Gnostic Local Distribution
    Function (GLDF), specifically `ELDF` or `QLDF`. Instead of classical residual
    tests, it relies on the gnostic scale parameter estimated by the GLDF.

    Key differences vs. standard tests:
    - Uses GLDF's scale parameter instead of residual-based heuristics.
    - Works with local/global gnostic scale (`S_local`, `S_opt`) for inference.
    - Integrates directly with Machine Gnostics models (`ELDF`, `QLDF`).

    Usage overview:
    - Fit a GLDF (`ELDF` or `QLDF`) with `varS=True` on your data.
    - Pass the fitted GLDF instance to `DataScedasticity`.
    - Call `fit()` to classify scedasticity and populate `result()`.

    Attributes:
    - `gldf`: GLDF instance (`ELDF` or `QLDF`) already fitted with `varS=True`.
    - `catch`: Whether to catch/handle warnings or errors (reserved for future).
    - `verbose`: Enables debug-level logging when `True`.
    - `params`: Results container populated after `fit()`.
    - `fitted`: Boolean indicating whether scedasticity classification ran.
    - `logger`: Module logger configured by `get_logger()`.

    Example:
    ```python
    from machinegnostics.magcal import ELDF, DataScedasticity

    # Prepare and fit GLDF with variable scale enabled
    eldf = ELDF(varS=True)
    eldf.fit(data)  # ensure the GLDF is fitted and exposes S_var, S_local, S_opt

    # Run scedasticity test
    sc = DataScedasticity(gldf=eldf, verbose=True)
    is_homo = sc.fit()
    info = sc.result()

    print(is_homo)           # True if homoscedastic, False otherwise
    print(info['scedasticity'])  # 'homoscedastic' or 'heteroscedastic'
    print(info['S_global'])  # global/optimal gnostic scale (S_opt)
    ```

    Note:
    - The supplied `gldf` must be an instance of `ELDF` or `QLDF`, fitted with
        `varS=True`. Validation is performed during initialization.
    - `fit()` sets `params` and `fitted`; call `result()` afterwards to obtain a
        structured dictionary of outputs.
    """

    def __init__(self,
                 gldf: Union[ELDF, QLDF] = ELDF,
                 catch: bool = True,
                 verbose: bool = False):
        """Initialize a scedasticity checker with a fitted GLDF.

        Args:
        - `gldf`: A fitted GLDF instance (`ELDF` or `QLDF`) configured with
            `varS=True` and exposing `S_var`, `S_local`, and `S_opt`.
        - `catch`: Reserved flag for future error/exception handling.
        - `verbose`: If `True`, enables debug-level logging for this checker.

        Raises:
        - `TypeError`: If `gldf` is not an `ELDF` or `QLDF` type.
        - `ValueError`: If `gldf` is not fitted or `varS` is `False`.

        Example:
        ```python
        from machinegnostics.magcal import ELDF, DataScedasticity

        eldf = ELDF(varS=True)
        eldf.fit(data)
        sc = DataScedasticity(gldf=eldf, verbose=True)
        ```
                """
        self.gldf = gldf
        self.catch = catch
        self.verbose = verbose
        
        self.params = {}
        self.fitted = False

        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")

        self._input_qldf_check()
        self._gdf_obj_validation()
    
    def _input_qldf_check(self):
        """Validate the input GLDF type is either `ELDF` or `QLDF`.

        This check ensures the provided `gldf` conforms to supported gnostic
        local distribution function types.

        Raises:
        - `TypeError`: If `gldf` is not an `ELDF` or `QLDF` type.

        Example:
        ```python
        sc = DataScedasticity(gldf=eldf)
        # Internal validation runs during initialization; you can call manually:
        sc._input_qldf_check()
        ```
        """
        self.logger.info("Validating input GLDF (Gnostic Local Distribution Function) class...")
        class_name = self.gldf.__class__.__name__
        if class_name not in ['ELDF', 'QLDF']:
            self.logger.error(f"Input GLDF class must be ELDF or QLDF, got {class_name} instead.")
            raise TypeError(f"Input GLDF class must be ELDF or QLDF, got {class_name} instead.")
        self.logger.info("Input GLDF class is valid.")
    
    def _gdf_obj_validation(self):
        """Validate `gldf` object is fitted and configured with `varS=True`.

        Ensures downstream scedasticity analysis can access `S_var`, `S_local`,
        and `S_opt` from the GLDF instance.

        Raises:
        - `ValueError`: If the GLDF is not fitted or `varS` is `False`.

        Example:
        ```python
        sc = DataScedasticity(gldf=eldf)
        sc._gdf_obj_validation()  # no-op if configuration is valid
        ```
        """
        self.logger.info("Validating GLDF object...")
        if not self.gldf._fitted:
            self.logger.error("The GLDF object must be fitted before checking scedasticity.")
            raise ValueError("The GLDF object must be fitted before checking scedasticity.")
        if not self.gldf.varS:
            self.logger.error("The GLDF object must have varS=True to check for scedasticity.")
            raise ValueError("The GLDF object must have varS=True to check for scedasticity.")
        self.logger.info("GLDF object is valid for scedasticity check.")


    def fit(self) -> bool:
        """Classify scedasticity using GLDF scale variability.

        Checks whether the GLDF's scale parameter array (`S_var`) is constant
        across data points. If constant, the data is classified as
        homoscedastic; otherwise, heteroscedastic.

        Returns:
        - `True`: Data is homoscedastic (constant scale parameter).
        - `False`: Data is heteroscedastic (variable scale parameter).

        Side effects:
        - Populates `self.params` with keys: `scedasticity`, `scale_parameter`,
          `S_local`, and `S_global`.
        - Sets `self.fitted = True` when complete.

        Example:
        ```python
        sc = DataScedasticity(gldf=eldf)
        is_homo = sc.fit()
        if is_homo:
            print("Homoscedastic")
        else:
            print("Heteroscedastic")
        ```
        """
        # get scale parameter S from gldf
        s_var = self.gldf.S_var

        # check if scale parameter is constant or variable
        if np.allclose(s_var, s_var[0]):
            self.logger.info("Data is homoscedastic (constant scale parameter).")
            if self.catch:
                self.params['scedasticity'] = 'homoscedastic'
                self.params['scale_parameter'] = s_var
                self.params['S_local'] = self.gldf.S_local
                self.params['S_global'] = self.gldf.S_opt
            self.fitted = True
            return True
        else:
            self.logger.info("Data is heteroscedastic (variable scale parameter).")
            if self.catch:
                self.params['scedasticity'] = 'heteroscedastic'
                self.params['scale_parameter'] = s_var
                self.params['S_local'] = self.gldf.S_local
                self.params['S_global'] = self.gldf.S_opt
            self.fitted = True
            return False

    def results(self) -> dict:
        """Return scedasticity results after `fit()`.

        Returns:
        - `dict`: A dictionary containing:
          - `scedasticity`: `'homoscedastic'` or `'heteroscedastic'`.
          - `scale_parameter`: The `S_var` array from the GLDF.
          - `S_local`: Local gnostic scale values.
          - `S_global`: Global/optimal gnostic scale (`S_opt`).

        Raises:
        - `ValueError`: If `fit()` has not been called.

        Example:
        ```python
        sc = DataScedasticity(gldf=eldf)
        sc.fit()
        info = sc.result()
        print(info['scedasticity'])  # 'homoscedastic' or 'heteroscedastic'
        ```
        """
        if not self.fitted:
            self.logger.error("The model must be fitted before retrieving results.")
            raise ValueError("The model must be fitted before retrieving results.")
        
        return self.params

    def __repr__(self):
        """Return a concise representation of the scedasticity checker.

        Intended to help with debugging/logging. Representation may include
        the GLDF type and `fitted` state in a future implementation.

        Example:
        ```python
        sc = DataScedasticity(gldf=eldf)
        repr(sc)  # e.g., '<DataScedasticity(gldf=ELDF, fitted=False)>'
        ```
        """
        try:
            gldf_type = self.gldf.__name__
        except AttributeError:
            gldf_type = self.gldf.__class__.__name__

        status = 'fitted' if self.fitted else 'unfitted'
        sced = self.params.get('scedasticity', 'unknown') if self.fitted else 'unknown'
        return f"<DataScedasticity(gldf={gldf_type}, {status}, scedasticity={sced})>"
