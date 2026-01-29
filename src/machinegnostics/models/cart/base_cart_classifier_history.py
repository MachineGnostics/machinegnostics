'''
History class for the Gnostic CART Classification model.

Machine Gnostics
'''

from machinegnostics.models.cart.base_cart_classifier_cal import CartClassifierCalBase
import logging
from machinegnostics.magcal.util.logging import get_logger

class HistoryCartClassifierBase(CartClassifierCalBase):
    """
    History class for Gnostic CART.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info("HistoryCartClassifierBase initialized.")
