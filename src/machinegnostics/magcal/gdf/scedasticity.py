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
    or heteroscedastic (varying scale) using a fitted Gnostic (Local) Distribution
    Function (GDF), specifically `ELDF` or `QLDF`. Instead of classical residual
    tests, it relies on the gnostic scale parameter estimated by the GDF.

    Key differences vs. standard tests:
    - Uses GDF's scale parameter instead of residual-based heuristics.
    - Works with local/global gnostic scale (`S_local`, `S_opt`) for inference.
    - Integrates directly with Machine Gnostics models (`ELDF`, `QLDF`).

    Usage overview:
    - Fit a GDF (`ELDF` or `QLDF`) with `varS=True` on your data.
    - Pass the fitted GDF instance to `DataScedasticity`.
    - Call `fit()` to classify scedasticity and populate `result()`.

    Attributes:
    - `gdf`: GDF instance (`ELDF` or `QLDF`) already fitted with `varS=True`.
    - `catch`: Whether to catch/handle warnings or errors (reserved for future).
    - `verbose`: Enables debug-level logging when `True`.
    - `params`: Results container populated after `fit()`.
    - `fitted`: Boolean indicating whether scedasticity classification ran.
    - `logger`: Module logger configured by `get_logger()`.

    Example:
    ```python
    from machinegnostics.magcal import ELDF, DataScedasticity

    # Prepare and fit GDF with variable scale enabled
    eldf = ELDF(varS=True)
    eldf.fit(data)  # ensure the GDF is fitted and exposes S_var, S_local, S_opt

    # Run scedasticity test
    sc = DataScedasticity(gdf=eldf, verbose=True)
    is_homo = sc.fit()
    info = sc.result()

    print(is_homo)           # True if homoscedastic, False otherwise
    print(info['scedasticity'])  # 'homoscedastic' or 'heteroscedastic'
    print(info['S_global'])  # global/optimal gnostic scale (S_opt)
    ```

    Methods:
    - `fit()`: Performs the scedasticity classification based on the fitted GDF.
    - `result()`: Returns a dictionary summarizing the scedasticity analysis.

    Note:
    - The supplied `gdf` must be an instance of `ELDF` or `QLDF`, fitted with
        `varS=True`. Validation is performed during initialization.
    - `fit()` sets `params` and `fitted`; call `result()` afterwards to obtain a
        structured dictionary of outputs.
    """

    def __init__(self,
                 gdf: Union[ELDF, QLDF] = ELDF,
                 catch: bool = True,
                 verbose: bool = False):
        """Initialize a scedasticity checker with a fitted GDF.

        Args:
        - `gdf`: A fitted GDF instance (`ELDF` or `QLDF`) configured with
            `varS=True` and exposing `S_var`, `S_local`, and `S_opt`.
        - `catch`: Reserved flag for future error/exception handling.
        - `verbose`: If `True`, enables debug-level logging for this checker.

        Raises:
        - `TypeError`: If `gdf` is not an `ELDF` or `QLDF` type.
        - `ValueError`: If `gdf` is not fitted or `varS` is `False`.

        Example:
        ```python
        from machinegnostics.magcal import ELDF, DataScedasticity

        eldf = ELDF(varS=True)
        eldf.fit(data)
        sc = DataScedasticity(gdf=eldf, verbose=True)
        ```
                """
        self.gdf = gdf
        self.catch = catch
        self.verbose = verbose
        
        self.params = {}
        self.fitted = False

        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")

        self._input_qldf_check()
        self._gdf_obj_validation()
    
    def _input_qldf_check(self):
        """Validate the input GDF type is either `ELDF` or `QLDF`.

        This check ensures the provided `gdf` conforms to supported gnostic
        local distribution function types.

        Raises:
        - `TypeError`: If `gdf` is not an `ELDF` or `QLDF` type.

        Example:
        ```python
        sc = DataScedasticity(gdf=eldf)
        # Internal validation runs during initialization; you can call manually:
        sc._input_qldf_check()
        ```
        """
        self.logger.info("Validating input GDF (Gnostic Local Distribution Function) class...")
        class_name = self.gdf.__class__.__name__
        if class_name not in ['ELDF', 'QLDF']:
            self.logger.error(f"Input GDF class must be ELDF or QLDF, got {class_name} instead.")
            raise TypeError(f"Input GDF class must be ELDF or QLDF, got {class_name} instead.")
        self.logger.info("Input GDF class is valid.")
    
    def _gdf_obj_validation(self):
        """Validate `gdf` object is fitted and configured with `varS=True`.

        Ensures downstream scedasticity analysis can access `S_var`, `S_local`,
        and `S_opt` from the GDF instance.

        Raises:
        - `ValueError`: If the GDF is not fitted or `varS` is `False`.

        Example:
        ```python
        sc = DataScedasticity(gdf=eldf)
        sc._gdf_obj_validation()  # no-op if configuration is valid
        ```
        """
        self.logger.info("Validating GDF object...")
        if not self.gdf._fitted:
            self.logger.error("The GDF object must be fitted before checking scedasticity.")
            raise ValueError("The GDF object must be fitted before checking scedasticity.")
        if not self.gdf.varS:
            self.logger.error("The GDF object must have varS=True to check for scedasticity.")
            raise ValueError("The GDF object must have varS=True to check for scedasticity.")
        self.logger.info("GDF object is valid for scedasticity check.")


    def fit(self) -> bool:
        """Classify scedasticity using GDF scale variability.

        Checks whether the GDF's scale parameter array (`S_var`) is constant
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
        sc = DataScedasticity(gdf=eldf)
        is_homo = sc.fit()
        if is_homo:
            print("Homoscedastic")
        else:
            print("Heteroscedastic")
        ```
        """
        # get scale parameter S from gldf
        s_var = self.gdf.S_var

        # check if scale parameter is constant or variable
        if np.allclose(s_var, s_var[0]):
            self.logger.info("Data is homoscedastic (constant scale parameter).")
            if self.catch:
                self.params['scedasticity'] = 'homoscedastic'
                self.params['scale_parameter'] = s_var
                self.params['S_local'] = self.gdf.S_local
                self.params['S_global'] = self.gdf.S_opt
            self.fitted = True
            return True
        else:
            self.logger.info("Data is heteroscedastic (variable scale parameter).")
            if self.catch:
                self.params['scedasticity'] = 'heteroscedastic'
                self.params['scale_parameter'] = s_var
                self.params['S_local'] = self.gdf.S_local
                self.params['S_global'] = self.gdf.S_opt
            self.fitted = True
            return False

    def results(self) -> dict:
        """Return scedasticity results after `fit()`.

        Returns:
                - `dict`: A dictionary containing:
          - `scedasticity`: `'homoscedastic'` or `'heteroscedastic'`.
                    - `scale_parameter`: The `S_var` array from the GDF.
          - `S_local`: Local gnostic scale values.
          - `S_global`: Global/optimal gnostic scale (`S_opt`).

        Raises:
        - `ValueError`: If `fit()` has not been called.

        Example:
        ```python
        sc = DataScedasticity(gdf=eldf)
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
        sc = DataScedasticity(gdf=eldf)
        repr(sc)  # e.g., '<DataScedasticity(gdf=ELDF, fitted=False)>'
        ```
        """
        try:
            gdf_type = self.gdf.__name__
        except AttributeError:
            gdf_type = self.gdf.__class__.__name__

        status = 'fitted' if self.fitted else 'unfitted'
        sced = self.params.get('scedasticity', 'unknown') if self.fitted else 'unknown'
        return f"<DataScedasticity(gdf={gdf_type}, {status}, scedasticity={sced})>"
