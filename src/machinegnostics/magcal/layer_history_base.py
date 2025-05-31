import numpy as np
from dataclasses import dataclass, asdict
from machinegnostics.magcal import ParamRobustRegressorBase

@dataclass
class HistoryRecord:
    iteration: int
    h_loss: float = None
    weights: np.ndarray = None
    coefficients: np.ndarray = None
    degree: int = None
    rentropy: float = None
    fi: np.ndarray = None
    hi: np.ndarray = None
    fj: np.ndarray = None
    hj: np.ndarray = None
    infoi: dict = None
    infoj: dict = None
    pi: np.ndarray = None
    pj: np.ndarray = None
    ei: float = None
    ej: float = None

class HistoryBase:
    """
    Base class for maintaining a history of model parameters and gnostic loss values.
    
    This class extends RegressorParamBase to provide functionality for tracking
    the evolution of model parameters and gnostic loss during training iterations.

    Parameters needed to record history:
        - h_loss: Gnostic loss value at each iteration
        - iteration: The iteration number
        - weights: Model weights at each iteration
        - coefficients: Model coefficients at each iteration
        - degree: Degree of polynomial features used in the model
        - rentropy: Entropy of the model at each iteration
        - fi - fi values, if calculated
        - hi - hi values, if calculated
        - fj - fj values, if calculated
        - hj - hj values, if calculated
        - infoi - gnostics information, if calculated
        - infoj - gnostics information, if calculated
        - pi - gnostic probabilities, if calculated
        - pj - gnostic probabilities, if calculated
        - ei - gnostic entropy, if calculated
        - ej - gnostic entropy, if calculated
    """
    
    def __init__(self, history: bool = True):
        """
        Initialize the HistoryBase class.

        Parameters
        ----------
        history : bool, default=True
            If True, enables recording of model parameters and gnostic loss history.
        """
        self._record_history = history
        if not isinstance(self._record_history, bool):
            raise ValueError("record_history must be a boolean value.")
        if self._record_history:
            self.history = []

    def record_history(
        self,
        iteration,
        h_loss=None,
        weights=None,
        coefficients=None,
        degree=None,
        rentropy=None,
        fi=None,
        hi=None,
        fj=None,
        hj=None,
        infoi=None,
        infoj=None,
        pi=None,
        pj=None,
        ei=None,
        ej=None
    ):
        """
        Record the current state of model parameters and metrics.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        h_loss : float, optional
            Gnostic loss value at this iteration.
        weights : np.ndarray, optional
            Model weights at this iteration.
        coefficients : np.ndarray, optional
            Model coefficients at this iteration.
        degree : int, optional
            Degree of polynomial features used.
        rentropy : float, optional
            Entropy of the model at this iteration.
        fi, hi, fj, hj : np.ndarray, optional
            fi, hi, fj, hj values, if calculated.
        infoi, infoj : dict, optional
            Gnostics information, if calculated.
        pi, pj : np.ndarray, optional
            Gnostic probabilities, if calculated.
        ei, ej : float, optional
            Gnostic entropy, if calculated.
        """
        if self._record_history:
            record = HistoryRecord(
                iteration=iteration,
                h_loss=h_loss,
                weights=np.copy(weights) if weights is not None else None,
                coefficients=np.copy(coefficients) if coefficients is not None else None,
                degree=degree,
                rentropy=rentropy,
                fi=np.copy(fi) if fi is not None else None,
                hi=np.copy(hi) if hi is not None else None,
                fj=np.copy(fj) if fj is not None else None,
                hj=np.copy(hj) if hj is not None else None,
                infoi=infoi.copy() if infoi is not None else None,
                infoj=infoj.copy() if infoj is not None else None,
                pi=np.copy(pi) if pi is not None else None,
                pj=np.copy(pj) if pj is not None else None,
                ei=ei,
                ej=ej
            )
            self.history.append(record)

    def get_history(self, as_dict=False):
        """
        Retrieve the recorded history.

        Parameters
        ----------
        as_dict : bool, default=False
            If True, returns a list of dictionaries. Otherwise, returns dataclass objects.

        Returns
        -------
        history : list
            List of HistoryRecord objects or dictionaries.
        """
        if as_dict:
            return [asdict(record) for record in self.history]
        return self.history

    def clear_history(self):
        """
        Clear the stored history.
        """
        self.history = []

    def prepare_history_for_output(self):
        """
        Prepare the history for output, e.g., for logging or saving.

        Returns
        -------
        output : dict
            Dictionary containing lists of each tracked attribute.
        """
        output = {
            "iteration": [],
            "h_loss": [],
            "weights": [],
            "coefficients": [],
            "degree": [],
            "rentropy": [],
            "fi": [],
            "hi": [],
            "fj": [],
            "hj": [],
            "infoi": [],
            "infoj": [],
            "pi": [],
            "pj": [],
            "ei": [],
            "ej": []
        }
        for record in self.history:
            output["iteration"].append(record.iteration)
            output["h_loss"].append(record.h_loss)
            output["weights"].append(record.weights)
            output["coefficients"].append(record.coefficients)
            output["degree"].append(record.degree)
            output["rentropy"].append(record.rentropy)
            output["fi"].append(record.fi)
            output["hi"].append(record.hi)
            output["fj"].append(record.fj)
            output["hj"].append(record.hj)
            output["infoi"].append(record.infoi)
            output["infoj"].append(record.infoj)
            output["pi"].append(record.pi)
            output["pj"].append(record.pj)
            output["ei"].append(record.ei)
            output["ej"].append(record.ej)
        return output

    @property
    def history(self):
        """
        Property to access the recorded history.
        """
        return self.get_history()