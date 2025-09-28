import numpy as np
from dataclasses import dataclass, asdict, field
import logging
from machinegnostics.magcal.util.logging import get_logger

@dataclass
class GenericHistoryRecord:
    # Store all attributes dynamically
    data: dict = field(default_factory=dict)

class HistoryBase:
    """
    Generic base class for maintaining a history of model parameters and metrics.
    Can be used for any model by recording arbitrary key-value pairs per iteration.
    """

    def __init__(self, history: bool = True):
        self._record_history = history
        self.logger = get_logger(self.__class__.__name__, logging.INFO)  # Create a logger for this class
        if not isinstance(self._record_history, bool):
            self.logger.error("record_history must be a boolean value.")
            raise ValueError("record_history must be a boolean value.")
        if self._record_history:
            self._history = []

        self.logger.info(f"HistoryBase initialized")

    def record_history(self, **kwargs):
        """
        Record the current state of model parameters and metrics.
        Accepts any keyword arguments to store as a record.
        """
        self.logger.info(f"Recording history: {kwargs}")
        if self._record_history:
            # Optionally, you can deepcopy arrays here if needed
            record = GenericHistoryRecord(data={k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in kwargs.items()})
            self._history.append(record)

    def get_history(self, as_dict=False):
        """
        Retrieve the recorded history.
        If as_dict=True, returns a list of dicts. Otherwise, returns dataclass objects.
        """
        self.logger.info("Retrieving history.")
        if as_dict:
            return [rec.data for rec in self._history]
        return self._history

    def clear_history(self):
        """Clear the stored history."""
        self.logger.info("Clearing history.")
        self._history = []

    def prepare_history_for_output(self):
        """
        Prepare the history for output as a dict of lists (for logging or saving).
        """
        self.logger.info("Preparing history for output.")
        if not self._history:
            return {}
        # Collect all unique keys
        all_keys = set()
        for rec in self._history:
            all_keys.update(rec.data.keys())
        output = {k: [] for k in all_keys}
        for rec in self._history:
            for k in all_keys:
                output[k].append(rec.data.get(k, None))
        return output

    @property
    def history(self):
        """Property to access the recorded history."""
        return self.get_history()