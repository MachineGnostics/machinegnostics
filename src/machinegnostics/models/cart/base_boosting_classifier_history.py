'''
History class for the Gnostic Boosting Classification model.

Machine Gnostics
'''

from machinegnostics.models.cart.base_boosting_classifier_cal import BoostingClassifierCalBase
import logging
from machinegnostics.magcal.util.logging import get_logger

class HistoryBoostingClassifierBase(BoostingClassifierCalBase):
    """
    History class for Gnostic Boosting Classifier.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info("HistoryBoostingClassifierBase initialized.")
