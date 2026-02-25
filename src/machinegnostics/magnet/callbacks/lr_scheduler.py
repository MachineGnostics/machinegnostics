"""LRScheduler callback for Magnet."""

class LRScheduler:
    def __init__(self, optimizer, factor=0.5, patience=5, min_lr=1e-5):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float('inf')
        self.wait = 0

    def step(self, current):
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                if new_lr < self.optimizer.lr:
                    print(f"Reducing LR to {new_lr}")
                    self.optimizer.lr = new_lr
                self.wait = 0
