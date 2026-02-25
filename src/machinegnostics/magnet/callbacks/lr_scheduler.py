
"""LRScheduler callback for Magnet."""
from .base import Callback

class LRScheduler(Callback):
    def __init__(self, optimizer, factor=0.5, patience=5, min_lr=1e-5, monitor='val_loss_0'):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.monitor = monitor
        self.best = float('inf')
        self.wait = 0

    def on_epoch_end(self, history, epoch, model):
        current = history[self.monitor][-1] if self.monitor in history and history[self.monitor] else None
        if current is None:
            return
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                if new_lr < self.optimizer.lr:
                    print(f"[LRScheduler] Reducing LR to {new_lr}")
                    self.optimizer.lr = new_lr
                self.wait = 0
