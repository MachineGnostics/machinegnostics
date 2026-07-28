"""Early stopping support for the basic Sequential stack."""

from __future__ import annotations

from copy import deepcopy


class EarlyStopping:
    def __init__(self, monitor="val_loss", patience=5, min_delta=0.0, restore_best_weights=True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best = None
        self.best_weights = None
        self.wait = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        self.best = None
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
            if self.restore_best_weights and self.model is not None:
                self.best_weights = deepcopy(self.model.get_weights())
            return

        self.wait += 1
        if self.wait >= self.patience and self.model is not None:
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        return logs