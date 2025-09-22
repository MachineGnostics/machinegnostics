'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar
'''
import numpy as np


class GnosticCriterion:
    """
    A class to compute Gnostic Error Functions for Machine Gnostics Regression.
    Gnostic Error Functions for Machine Gnostics Regression.
    Reference: TABLE 19.2, p. 19-4, in the provided reference.
    """

    @staticmethod
    def _compute_error(case:str, f_j=None, f_i=None, h_j=None, h_i=None):
        """
        Compute the error function based on the specified case.

        Args:
            case (str): The case to compute the error for ('Q1', 'E1', 'Q2', 'E2', 'Q3', 'E3').
            f_j (float): Filtering weight for J cases.
            f_i (float): Filtering weight for I cases.
            h_j (float): Error function input for J cases.
            h_i (float): Error function input for I cases.

        Returns:
            float: The computed error function value.
        """
        if case == 'Q1':
            if f_j is None:
                raise ValueError("f_j is required for case Q1")
            return f_j  # h_j = f_j

        elif case == 'E1':
            if f_i is None:
                raise ValueError("f_i is required for case E1")
            return f_i  # h_i = f_i

        elif case == 'Q2':
            if f_j is None or h_j is None:
                raise ValueError("f_j and h_j are required for case Q2")
            return f_j * np.arctan(h_j)  # f_j * arctan(h_j)

        elif case == 'E2':
            if f_i is None or h_i is None:
                raise ValueError("f_i and h_i are required for case E2")
            return f_i * np.arctanh(h_i)  # f_i * arctanh(h_i)

        elif case == 'Q3':
            if f_j is None or h_j is None:
                raise ValueError("f_j and h_j are required for case Q3")
            return np.sqrt(f_j) * h_j  # sqrt(f_j) * h_j

        elif case == 'E3':
            if f_i is None or h_i is None:
                raise ValueError("f_i and h_i are required for case E3")
            return np.sqrt(f_i) * h_i  # sqrt(f_i) * h_i

        else:
            raise ValueError(f"Invalid case: {case}. Valid cases are 'Q1', 'E1', 'Q2', 'E2', 'Q3', 'E3'.")

    @staticmethod
    def _get_filtering_weight(case:str, f_j=None, f_i=None):
        """
        Calculate the filtering weight based on the specified case.

        Args:
            case (str): The case to compute the filtering weight for ('Q1', 'E1', 'Q2', 'E2', 'Q3', 'E3').
            f_j (float): Filtering weight for J cases.
            f_i (float): Filtering weight for I cases.

        Returns:
            float: The computed filtering weight.
        """
        if case == 'Q1':
            if f_j is None:
                raise ValueError("f_j is required for case Q1")
            return f_j  # Filtering weight is f_j

        elif case == 'E1':
            if f_i is None:
                raise ValueError("f_i is required for case E1")
            return f_i**2  # Filtering weight is f_i^2

        elif case == 'Q2':
            return 1  # Filtering weight is 1

        elif case == 'E2':
            if f_i is None:
                raise ValueError("f_i is required for case E2")
            return f_i  # Filtering weight is f_i

        elif case == 'Q3':
            if f_j is None:
                raise ValueError("f_j is required for case Q3")
            return 1 / np.sqrt(f_j)  # Filtering weight is 1 / sqrt(f_j)

        elif case == 'E3':
            if f_i is None:
                raise ValueError("f_i is required for case E3")
            return 1 / np.sqrt(f_i)  # Filtering weight is 1 / sqrt(f_i)

        else:
            raise ValueError(f"Invalid case: {case}. Valid cases are 'Q1', 'E1', 'Q2', 'E2', 'Q3', 'E3'.")
        
    @staticmethod
    def _get_gnostic_criterion(case:str, f_j=None, f_i=None, h_j=None, h_i=None, Ii=None, Ij=None):
        """
        Get the gnostic criterion based on the specified case.

        Args:
            case (str): The case to compute the gnostic criterion for ('Q1', 'E1', 'Q2', 'E2', 'Q3', 'E3').
            f_j (float): Filtering weight for J cases.
            f_i (float): Filtering weight for I cases.
            h_j (float): Error function input for J cases.
            h_i (float): Error function input for I cases.
            Ii (float): Estimating information.
            Ij (float): Quantifying information.

        Returns:
            float: The computed gnostic criterion value.

        """
        if case == 'Q1':
            return h_i**2 / 2
        elif case == 'E1':
            return h_j**2 / 2
        elif case == 'Q2':
            return Ij
        elif case == 'E2':
            return Ii
        elif case == 'Q3':
            return f_j
        elif case == 'E3':
            return -f_i

        
