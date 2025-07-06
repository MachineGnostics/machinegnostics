"""
EGDF - Estimating Global Distribution Function.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
import warnings
from machinegnostics.magcal.scale_param import ScaleParam
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.gdf.base_df import BaseDistFunc
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.gdf.wedf import WEDF
from machinegnostics.magcal.gdf.homogeneity import DataHomogeneity

class EGDF(BaseDistFunc, DataHomogeneity):
    """
    EGDF - A class for estimating the global distribution function.
    """
    def __init__(self,
                 data: np.ndarray,
                 DLB: float = None,
                 DUB: float = None,
                 LSB: float = None,
                 USB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 tolerance: float = 1e-3,
                 data_form: str = 'a',
                 n_points: int = 100,
                 homogeneous: bool = True,
                 catch: bool = True,
                 weights: np.ndarray = None):
        
        """
        Initialize the EGDF class.

        Parameters:
        data (np.ndarray): Input data for the EGDF.
        DLB (float): Lower bound for the data.
        DUB (float): Upper bound for the data.
        LSB (float): Lower Sample Bound.
        USB (float): Upper Sample Bound.
        LB (float): Lower (Probable) Bound.
        UB (float): Upper (Probable) Bound.
        S (float): Scale parameter.
        tolerance (float): Tolerance for convergence.
        data_form (str): Form of the data ('a' for additive, 'm' for multiplicative).
        n_points (int): Number of points in the distribution function.
        homogeneous (bool): Whether given data is homogeneous, True by default. if False, in that case data will be homogenized.
        catch (bool): To catch calculated values or not, True by default.
        weights (np.ndarray): Priory Weights for the data points.
        data_pad (float): Padding for the data range.
        """
        
        self.data = data
        self.DLB = DLB
        self.DUB = DUB
        self.LSB = LSB
        self.USB = USB
        self.LB = LB
        self.UB = UB
        self.S = S
        self.tolerance = tolerance
        self.data_form = data_form
        self.n_points = n_points
        self.homogeneous = homogeneous
        self.catch = catch
        self.weights = weights if weights is not None else np.ones_like(data)
        self.params = {}
        
        # argument validation
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array.")
        
        # Validate only if not None
        if DLB is not None and not isinstance(DLB, (int, float)):
            raise ValueError("DLB must be a numeric value or None.")
        if DUB is not None and not isinstance(DUB, (int, float)):
            raise ValueError("DUB must be a numeric value or None.")
        if LSB is not None and not isinstance(LSB, (int, float)):
            raise ValueError("LSB must be a numeric value or None.")
        if USB is not None and not isinstance(USB, (int, float)):
            raise ValueError("USB must be a numeric value or None.")
        if LB is not None and not isinstance(LB, (int, float)):
            raise ValueError("LB must be a numeric value or None.")
        if UB is not None and not isinstance(UB, (int, float)):
            raise ValueError("UB must be a numeric value or None.")
        
        # S can be int, float, or str 'auto' to automatically determine the scale parameter
        if not isinstance(S, (int, float, str)):
            raise ValueError("S must be a numeric positive value or 'auto'.")
        
        if not isinstance(tolerance, (int, float)):
            raise ValueError("Tolerance must be a numeric value.")
        
        if data_form not in ['a', 'm']:
            raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative.")
        
        if not isinstance(n_points, int) or n_points <= 0:
            raise ValueError("n_points must be a positive integer.")
        
        if not isinstance(homogeneous, bool):
            raise ValueError("homogeneous must be a boolean value. It indicates whether the input data is homogeneous or not.")
        
        if not isinstance(catch, bool):
            raise ValueError("catch must be a boolean value. It indicates whether to catch calculated values or not.")

        if weights is not None and not isinstance(weights, np.ndarray):
            raise ValueError("weights must be a numpy array.")
        if weights is not None and len(weights) != len(data):
            raise ValueError("Weights must have the same length as data.")
        
        # initial estimates and processing
        # data sorting
        self.data = np.sort(data)
        # DLB, DUB bounds
        self._estimate_data_bounds()
        # weights validation
        self._estimate_weights()

        # store if arguments are available
        if self.catch:
            self.params['DLB'] = self.DLB if self.DLB is not None else None
            self.params['DUB'] = self.DUB if self.DUB is not None else None
            self.params['LSB'] = self.LSB if self.LSB is not None else None
            self.params['USB'] = self.USB if self.USB is not None else None
            self.params['LB'] = self.LB if self.LB is not None else None
            self.params['UB'] = self.UB if self.UB is not None else None
            self.params['S'] = self.S if self.S is not None else None
            self.params['tolerance'] = self.tolerance if self.tolerance is not None else None
            self.params['data_form'] = self.data_form if self.data_form is not None else None
            self.params['n_points'] = self.n_points if self.n_points is not None else None
            self.params['homogeneous'] = self.homogeneous if self.homogeneous is not None else None
            self.params['weights'] = self.weights if self.weights is not None else None
        else:
            self.params = {}

        # initialize parent class
        DataHomogeneity.__init__(self, data=self.data, params=self.params)

    def _estimate_weights(self):
        """
        Estimate weights for the EGDF.
        
        This method can be overridden in subclasses to provide custom weight estimation logic.
        """
        # Default implementation uses uniform weights
        if self.weights is None:
            self.weights = np.ones_like(self.data)
        else:
            self.weights = np.asarray(self.weights)
            if len(self.weights) != len(self.data):
                raise ValueError("weights must have the same length as data")
        # Normalize weights to sum to n (number of data points)
        self.weights = self.weights / np.sum(self.weights) * len(self.weights)

        # store weights to param
        if self.catch:
            self.params['weights'] = self.weights
        else:
            self.params['weights'] = None

    def _estimate_data_bounds(self):
        """
        Estimate data bounds based on the EGDF.

        DLB and DUB are the data bounds where samples are expected.
        """
        # Estimate data bounds
        if self.DLB is None:
            self.DLB = np.min(self.data)
        if self.DUB is None:
            self.DUB = np.max(self.data)
        if self.catch:
            self.params['DLB'] = self.DLB
            self.params['DUB'] = self.DUB
        else:
            self.params['DLB'] = None
            self.params['DUB'] = None

    def _estimate_probable_bounds(self):
        """
        Estimate probable bounds based on the EGDF.

        LB and UB are the probable bounds where samples are expected.
        """
        if self.LB is None:
            self.LB = np.min(self.z)
        if self.UB is None:
            self.UB = np.max(self.z)
        if self.catch:
            self.params['LB_init'] = self.LB
            self.params['UB_init'] = self.UB
        else:
            self.params['LB_init'] = None
            self.params['UB_init'] = None

    def fit(self):
        """
        Fit the EGDF model to the data.

        This method applies the transformation and prepares the data for further analysis.
        """
        # transform data from standard domain to normal domain
        self._tranform_input()

        # R = Zi_points/Zi, calculate q, q1, and fidelity and irrelevance at S=1
        self._get_z_points()
        
        # estimate wedf
        self.wedf = self._get_wedf()

        # # estimate EGDF at S=1, fidelity, and irrelevance
        self.df = self._get_egdf()

        # find optimized bounds
        self._find_optimized_bounds()

        # R = egdf / wedf, calculate q, q1, and fidelity and irrelevance at S=1
        eps = np.finfo(float).eps  # small value to avoid division by zero
        R_df = self.df / (self.wedf + eps)
        gc = GnosticsCharacteristics(R=R_df)
        sp = ScaleParam()

        # if S is int or float, then calculate q, q1, fi, hi
        if isinstance(self.S, (int, float)):
            q, q1 = gc._get_q_q1(S=self.S)
            fi = gc._fi(q=q, q1=q1)
            hi = gc._hi(q=q, q1=q1)

        elif isinstance(self.S, str) and self.S.lower() == 'auto':
            # if S is 'auto', then calculate S using ScaleParam
            q, q1 = gc._get_q_q1(S=1)
            fi = gc._fi(q=q, q1=q1)
            hi = gc._hi(q=q, q1=q1)

            # S optimization
            self.S_opt_df = sp._gscale_loc(np.mean(fi))
            q, q1 = gc._get_q_q1(S=self.S_opt_df)
            fi = gc._fi(q=q, q1=q1)
            hi = gc._hi(q=q, q1=q1)
        
        # if any value in fi is Nan or infinite, then replace with 0
        if np.any(np.isnan(fi)) or np.any(np.isinf(fi)):
            fi = np.nan_to_num(fi, nan=0, posinf=0, neginf=0)
            hi = np.nan_to_num(hi, nan=0, posinf=0, neginf=0)
        
        # store fidelity, irrelevance, and S_opt to params
        if self.catch:
            self.params['fidelity'] = fi
            self.params['irrelevance'] = hi
            self.params['S_df'] = self.S_opt_df

        # calculate final EGDF with optimized S
        self.egdf =self._estimate_egdf(fi, hi)

        # find optimized bounds
        # self._find_optimized_bounds()

        # calculate final PDF
        self.pdf = self._get_pdf()

        # if data was not homogeneous, and checking that is homogenized or not
        if self.homogeneous == False:
            self._is_homogeneous()
            # if self.is_homo == False:
            #     raise Warning("Please check data homogeneity.")

    def plot(self):
        """
        plot EGDF and PDF.

        EGDF is in blue line on y1 axis and PDF is in red line on y2 axis.
        Optimized bounds LB and UB are shown as vertical lines.
        """
        import matplotlib.pyplot as plt

        if self.params.get('egdf') is None or self.params.get('pdf') is None:
            raise ValueError("EGDF and PDF must be calculated before plotting.")

        egdf = self.egdf_values
        pdf = self.pdf
        # Use the original di_points (data domain) for plotting instead of zi_points
        x_points = self.di_points

        fig, ax1 = plt.subplots()

        # Plot EGDF
        ax1.plot(x_points, egdf, color='blue', label='EGDF', linewidth=2)
        ax1.set_xlabel('Data Points')
        ax1.set_ylabel('EGDF', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        # Most robust version
        lb = self.params['LB']
        ub = self.params['UB']

        # Calculate range and apply 2% padding
        range_val = ub - lb
        padding = range_val * 0.02

        x_min = lb - padding
        x_max = ub + padding

        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(0, 1)

        # Create a second y-axis for the PDF
        ax2 = ax1.twinx()
        ax2.plot(x_points, pdf, color='red', label='PDF', linewidth=2)
        ax2.set_ylabel('PDF', color='red')
        ax2.set_ylim(0, np.max(pdf) * 1.1)  # Adjust y-limits for PDF
        ax2.tick_params(axis='y', labelcolor='red')

        # Add data points as vertical lines
        for point in self.data:
            ax1.axvline(x=point, color='gray', linestyle='--', alpha=0.5)

        # Add optimized bounds if available
        if self.params.get('LB') is not None:
            ax1.axvline(x=self.params['LB'], color='black', linestyle='--', linewidth=1.5, 
                    alpha=0.7)
            # Add pink shaded area to the left of LB
            ax1.axvspan(x_points.min(), self.params['LB'], alpha=0.2, color='pink', label=f"LB={self.params['LB']:.3f}")

        if self.params.get('UB') is not None:
            ax1.axvline(x=self.params['UB'], color='black', linestyle='--', linewidth=1.5, 
                    alpha=0.7)
            # Add green shaded area to the right of UB
            ax1.axvspan(self.params['UB'], x_points.max(), alpha=0.2, color='lightgreen', label=f"UB={self.params['UB']:.3f}")

        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title('EGDF and PDF')
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
    
    def _tranform_input(self):
        """
        Transform input data to the standard domain.

        This method can be overridden in subclasses to provide custom transformation logic.
        """
        dc = DataConversion()

        if self.data_form == 'a':
            self.z = dc._convert_az(self.data, self.DLB, self.DUB)
        elif self.data_form == 'm':
            self.z = dc._convert_mz(self.data, self.DLB, self.DUB)
        else:
            raise ValueError(f"Unknown data form: {self.data_form}, expected 'a' or 'm'.")
                
        # initial bounds
        self._estimate_probable_bounds()
        # infinite domain
        self.zi = dc._convert_fininf(self.z, self.LB, self.UB)
        # store
        if self.catch:
            self.params['z'] = self.z
            self.params['zi'] = self.zi
        else:
            self.params['z'] = None
            self.params['zi'] = None

        # homogenize
        if not self.homogeneous:
            self._homogenize()
    
    def _get_z_points(self):
        """
        Get the z points for the EGDF.

        Returns:
            np.ndarray: The z points.
        """
        if self.data_form == 'a':
            pad = (self.DUB - self.DLB) / 2
            self.DLB_ = self.DLB - pad
            self.DUB_ = self.DUB + pad
        elif self.data_form == 'm':
            self.DLB_ = self.DLB / np.sqrt(self.DUB / self.DLB)
            self.DUB_ = self.DUB * np.sqrt(self.DUB / self.DLB)

        # NOTE in future this logic can be improved to handle more complex cases
        # generate di_points
        self.di_points = np.linspace(self.DLB_, self.DUB_, self.n_points)

        # data transformation
        dc = DataConversion()
        if self.data_form == 'a':
            self.z_points = dc._convert_az(self.di_points, self.DLB, self.DUB)
        elif self.data_form == 'm':
            self.z_points = dc._convert_mz(self.di_points, self.DLB, self.DUB)
        else:
            raise ValueError(f"Unknown data form: {self.data_form}, expected 'a' or 'm'.")
        # finite to infinite domain conversion
        zi_points = dc._convert_fininf(self.z_points, self.LB, self.UB)

        # reshape
        self.zi_points = zi_points.reshape(-1, 1) if zi_points.ndim == 1 else zi_points

        if self.catch:
            self.params['di_points'] = self.di_points
            self.params['z_points'] = self.z_points
            self.params['zi_points'] = self.zi_points
        else:
            self.params['zi_points'] = None

    def _get_wedf(self):
        """
        Get the WEDF for the EGDF.

        Returns:
            WEDF: The WEDF object.
        """
        wedf = WEDF(self.data, weights=self.weights, data_lb=self.DLB, data_ub=self.DUB)
        
        wedf_values = wedf.fit(self.di_points)
        
        if self.catch:
            self.params['wedf'] = wedf_values
        else:
            self.params['wedf'] = None
        return wedf_values

    def _get_egdf(self):
        """
        Get the EGDF for the given scale parameter S.
        """
        eps = np.finfo(float).eps
        
        # R calculation for each zi_point against all data points
        zi_data = self.zi.reshape(-1, 1)  # Shape: (n_data, 1)
        zi_points_broadcast = self.zi_points.reshape(1, -1)  # Shape: (1, n_points)
        
        # R matrix: each row is data point, each column is evaluation point
        R = zi_data / (zi_points_broadcast + eps)
        
        gc = GnosticsCharacteristics(R=R)
        
        # if self.s is int or float, then calculate q, q1, fi, hi
        if isinstance(self.S, (int, float)):
            q, q1 = gc._get_q_q1(S=self.S)
            fi = gc._fi(q=q, q1=q1)
            hi = gc._hi(q=q, q1=q1)

        if isinstance(self.S, str) and self.S.lower() == 'auto':
            # if S is 'auto', then calculate S using ScaleParam
            sp = ScaleParam()
            q, q1 = gc._get_q_q1(S=1)
            fi = gc._fi(q=q, q1=q1)
            hi = gc._hi(q=q, q1=q1)

            # if any value in fi is Nan or infinite, then replace with 0
            if np.any(np.isnan(fi)) or np.any(np.isinf(fi)):
                fi = np.nan_to_num(fi, nan=0, posinf=0, neginf=0)
                hi = np.nan_to_num(hi, nan=0, posinf=0, neginf=0)

            # S optimization
            self.S_opt = sp._gscale_loc(np.mean(fi))
            if self.S_opt is None:
                self.S_opt = 1
                raise Warning("S_opt is None, using S=1 as default.")
            q, q1 = gc._get_q_q1(S=self.S_opt)
            fi = gc._fi(q=q, q1=q1)
            hi = gc._hi(q=q, q1=q1)
            
        # fi and hi are now calculated, store them
        # if any value in fi is Nan or infinite, then replace with 0
        if np.any(np.isnan(fi)) or np.any(np.isinf(fi)):
            fi = np.nan_to_num(fi, nan=0, posinf=0, neginf=0)
            hi = np.nan_to_num(hi, nan=0, posinf=0, neginf=0)

        self.fi = fi
        self.hi = hi

        if self.catch:
            self.params['S_opt'] = self.S_opt if self.S_opt is not None else 1
        else:
            self.params['S_opt'] = None

        # estimate egdf values
        self.egdf_values = self._estimate_egdf(fi, hi)
        
        return self.egdf_values

    def _estimate_egdf(self, fidelities, irrelevances):
        """
        Estimate the EGDF based on the fidelities and irrelevances.
        
        Parameters:
        fidelities : np.ndarray
            Fidelity values for each data point at each evaluation point
        irrelevances : np.ndarray
            Irrelevance values for each data point at each evaluation point
        
        Returns:
            np.ndarray: The estimated EGDF values.
        """
        # Calculate weighted means using equation 15.31
        weights = self.weights.reshape(-1, 1)  # Shape: (n_data, 1)
        
        mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)  # f̄_E
        mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)  # h̄_E
        
        # Calculate estimating modulus using equation 15.28
        M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
        
        # Avoid division by zero
        M_zi = np.where(M_zi == 0, np.finfo(float).eps, M_zi)
        
        # Calculate EGDF using equation 15.29
        egdf_values = (1 - mean_irrelevance / M_zi) / 2
        
        # Flatten the result to ensure it's 1D
        egdf_values = egdf_values.flatten()

        if self.catch:
            self.params['fidelity'] = mean_fidelity
            self.params['irrelevance'] = mean_irrelevance
            self.params['egdf'] = egdf_values
        else:
            self.params['egdf'] = None

        return egdf_values

    
    def _get_pdf(self):
        """
        Get the PDF for the EGDF using fidelity and irrelevance calculations.

        Returns:
            np.ndarray: The PDF values.
        """
        # Initialize output array with correct shape
        density = np.zeros_like(self.zi_points.flatten(), dtype=float)
        
        # Handle empty data case
        if len(self.zi) == 0:
            return density
        
        # Get fidelities and irrelevances from stored params
        fidelities = self.fi  # Use stored fi from _get_egdf
        irrelevances = self.hi  # Use stored hi from _get_egdf
        
        if fidelities is None or irrelevances is None:
            raise ValueError("Fidelities and irrelevances must be calculated before PDF estimation.")
        
        # Reshape weights for broadcasting
        weights = self.weights.reshape(-1, 1)  # Shape: (n_data, 1)
        
        # Calculate weighted means using equation 15.31
        mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)  # f̄_E
        mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)  # h̄_E

        # Calculate F2 and FH using equation 15.31
        F2 = np.sum(weights * fidelities**2, axis=0) / np.sum(weights)
        FH = np.sum(weights * fidelities * irrelevances, axis=0) / np.sum(weights)
        
        # Calculate estimating modulus using equation 15.28
        M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
        
        # Avoid division by zero
        eps = np.finfo(float).eps
        M_zi = np.where(M_zi == 0, eps, M_zi)
        M_zi_cubed = M_zi**3
        
        # Calculate PDF using equation 15.30
        self.sparam = self.S_opt if hasattr(self, 'S_opt') and self.S_opt is not None else 1.0
        numerator = (mean_fidelity**2) * F2 + mean_fidelity * mean_irrelevance * FH
        density = (1 / self.sparam) * (numerator / M_zi_cubed)
        
        # Flatten the result to ensure it's 1D
        density = density.flatten()
        
        # Handle negative density values
        if np.any(density < 0):
            warnings.warn("EGDF density contains negative values, which may indicate non-homogeneous data", RuntimeWarning)
        
        if self.catch:
            self.params['pdf'] = density
            # self.params['F2'] = F2
            # self.params['FH'] = FH
        else:
            self.params['pdf'] = None

        return density
    
    def _find_optimized_bounds(self, target_cdf_lower=0.00001, target_cdf_upper=0.99999):
        """
        Find optimized bounds using interpolation, ensuring bounds are outside original data range.
        
        Parameters:
        target_cdf_lower (float): Target CDF value for lower bound
        target_cdf_upper (float): Target CDF value for upper bound
        """
        egdf_values = self.params['egdf']
        di_points = self.params['di_points']
        data_min = self.data.min()
        data_max = self.data.max()
        
        # Interpolate to find exact points where EGDF = target values
        from scipy.interpolate import interp1d
        
        # Create interpolation function (EGDF -> di_points)
        # Remove any duplicate EGDF values for interpolation
        unique_indices = np.unique(egdf_values, return_index=True)[1]
        egdf_unique = egdf_values[unique_indices]
        di_unique = di_points[unique_indices]
        
        # Sort by EGDF values for interpolation
        sort_indices = np.argsort(egdf_unique)
        egdf_sorted = egdf_unique[sort_indices]
        di_sorted = di_unique[sort_indices]
        
        if len(egdf_sorted) > 1:
            interp_func = interp1d(egdf_sorted, di_sorted, 
                                bounds_error=False, fill_value='extrapolate')
            
            # Find bounds
            optimized_lower = float(interp_func(target_cdf_lower))
            optimized_upper = float(interp_func(target_cdf_upper))
            
            # Ensure bounds are outside data range
            # Lower bound must be < data.min()
            if optimized_lower >= data_min:
                # Find the rightmost di_point that is < data_min with lowest CDF
                valid_lower_points = di_points[di_points < data_min]
                if len(valid_lower_points) > 0:
                    # Get the index of the rightmost valid point
                    valid_lower_idx = np.where(di_points < data_min)[0][-1]
                    optimized_lower = di_points[valid_lower_idx]
                else:
                    # If no valid points, use minimum di_point
                    optimized_lower = di_points[0]
            
            # Upper bound must be > data.max()
            if optimized_upper <= data_max:
                # Find the leftmost di_point that is > data_max with highest CDF
                valid_upper_points = di_points[di_points > data_max]
                if len(valid_upper_points) > 0:
                    # Get the index of the leftmost valid point
                    valid_upper_idx = np.where(di_points > data_max)[0][0]
                    optimized_upper = di_points[valid_upper_idx]
                else:
                    # If no valid points, use maximum di_point
                    optimized_upper = di_points[-1]
                    
        else:
            # Fallback to edge values
            optimized_lower = di_points[0]
            optimized_upper = di_points[-1]
        
        # Store optimized bounds in params
        if self.catch:
            self.params['LB'] = optimized_lower
            self.params['UB'] = optimized_upper
        else:
            self.params['LB'] = None
            self.params['UB'] = None
    
    def _homogenize(self):
        """
        Homogenize the data if it is not homogeneous.
        
        This method can be overridden in subclasses to provide custom homogenization logic.
        """
        self.weights = self.homogenize()
        
    def _is_homogeneous(self):
        """
        Check if the data is homogeneous.
        
        This method can be overridden in subclasses to provide custom homogeneity checks.
        """
        self.is_homo, self.params = self.is_homogeneous()

        if self.catch:
            self.params['is_homogeneous'] = self.is_homo