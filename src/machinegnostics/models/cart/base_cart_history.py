'''
History class for the Gnostic CART model.

Machine Gnostics
'''

from machinegnostics.models.cart.base_cart_cal import CartCalBase
import logging
from machinegnostics.magcal.util.logging import get_logger

class HistoryCartBase(CartCalBase):
    """
    History class for Gnostic CART.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info("HistoryCartBase initialized.")
        
    # _fit is inherited from CartCalBase which already handles history
