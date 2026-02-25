"""Loss namespace for Magnet.

Author: Nirmal Parmar
"""

from .losses import BCELoss, CrossEntropyLoss, Loss, MAELoss, MSELoss

__all__ = ["Loss", "MSELoss", "MAELoss", "BCELoss", "CrossEntropyLoss"]
