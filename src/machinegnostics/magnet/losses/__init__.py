"""Loss functions for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This package re-exports the public MAGNET loss surface from dedicated modules
and keeps the flat API stable for user imports.
"""

from ._shared import Loss
from .binary_crossentropy import BinaryCrossEntropy
from .mse import MSE
from .gnostic_fidelity_loss import FidelityLoss
from .gnostic_infidelity_loss import InfidelityLoss
from .gnostic_rss_loss import RSSLoss
from .gnostic_iss_loss import ISSLoss
from .gnostic_information_loss import InformationLoss
from .residual_entropy_loss import ResidualEntropyLoss
from .gnostic_mse_loss import MSELoss
from .gnostic_binary_crossentropy_loss import BinaryCrossEntropyLoss
from .helpers import (
	gnostic_weighted_mse,
	gnostic_weighted_rmse,
	fidelity_loss,
	infidelity_loss,
	irrelevance_loss,
	relevance_loss,
	gnostic_characteristic_loss,
	get_loss,
)

__all__ = [
	"Loss", "BinaryCrossEntropy", "MSE", "FidelityLoss", "InfidelityLoss", "RSSLoss", "ISSLoss",
	"InformationLoss", "ResidualEntropyLoss", "MSELoss", "BinaryCrossEntropyLoss",
	"gnostic_weighted_mse", "gnostic_weighted_rmse", "fidelity_loss", "infidelity_loss",
	"irrelevance_loss", "relevance_loss", "gnostic_characteristic_loss", "get_loss",
]
