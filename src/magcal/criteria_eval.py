import numpy as np
from src.magcal.characteristics import GnosticsCharacteristics
from src.magcal.sample_characteristics import GnosticCharacteristicsSample

class CriteriaEvaluator:
    def __init__(self, y, y_fit, w=None):
        self.y = y
        self.y_fit = y_fit
        self.w = w
        self.robr2 = None
        self.gmmfe = None
        self.divI = None
        self.evalmet = None
        self._report = {}
        pass
    def _robr2():
        y = np.asarray(y)
        y_fit = np.asarray(y_fit)
        if w is None:
            w = np.ones_like(y)
        w = np.asarray(w)
        e = y - y_fit
        e_bar = np.sum(w * e) / np.sum(w)
        y_bar = np.sum(w * y) / np.sum(w)
        num = np.sum(w * (e - e_bar) ** 2)
        denom = np.sum(w * (y - y_bar) ** 2)
        return 1 - num / denom if denom != 0 else 0.0

    def _gmmfe():
        y = np.asarray(y)
        y_fit = np.asarray(y_fit)
        log_ratios = np.abs(np.log(y / y_fit))
        return np.exp(np.mean(log_ratios))

    def _divI(self, Iy, Iy_fit):
        gcs_y = GnosticCharacteristicsSample(data=self.y)
        gcs_y_fit = GnosticCharacteristicsSample(data=self.y_fit)

        y_median = gcs_y._gnostic_median(case='i')
        y_fit_median = gcs_y_fit._gnostic_median(case='i')

        # y/median
        zy = self.y / y_median
        zf = self.y_fit / y_fit_median

        # q
        gc_y = GnosticsCharacteristics(zy)
        gc_y_fit = GnosticsCharacteristics(zf)

        qy, q1y = gc_y._get_q_q1()
        qf, q1f = gc_y_fit._get_q_q1()

        # hi
        hi = gc_y._hi(q=qy, q1=q1y)
        hi_fit = gc_y_fit._hi(q=qf, q1=q1f)
        # pi
        pi = gc_y._idistfun(hi)
        pi_fit = gc_y_fit._idistfun(hi_fit)

        # Ii
        Iy = gc_y._info_i(pi)
        Iy_fit = gc_y_fit._info_i(pi_fit)

        return np.mean(Iy / Iy_fit)

    def _evalmet(robr2_val, gmmfe_val, divi_val):
        return robr2_val / (gmmfe_val * divi_val)