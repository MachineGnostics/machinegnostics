'''
History class for the Gnostic Boosting model.

Machine Gnostics
'''

from machinegnostics.models.cart.base_boosting_cal import BoostingCalBase
import logging
from machinegnostics.magcal.util.logging import get_logger

class HistoryBoostingBase(BoostingCalBase):
    """
    History class for Gnostic Boosting.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info("HistoryBoostingBase initialized.")
