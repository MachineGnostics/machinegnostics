
"""EarlyStopping callback for Magnet."""
from .base import Callback

class EarlyStopping(Callback):
    def __init__(self, patience=10, min_delta=1e-4, monitor='val_loss_0'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def on_epoch_end(self, history, epoch, model):
        current = history[self.monitor][-1] if self.monitor in history and history[self.monitor] else None
        if current is None:
            return
        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                self.stopped_epoch = epoch
