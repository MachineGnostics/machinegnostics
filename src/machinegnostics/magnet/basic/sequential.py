import numpy as np

from .early_stopping import EarlyStopping


class Sequential:
    def __init__(self, layers=None):
        self.layers = layers or []
        self.loss_fn = None
        self.optimizer = None
        self.history = {"loss": [], "val_loss": []}
        self.stop_training = False

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss_fn = loss
        self.optimizer = optimizer

    def get_weights(self):
        return [param.copy() for layer in self.layers for param in layer.params.values()]

    def set_weights(self, weights):
        index = 0
        for layer in self.layers:
            for key in layer.params:
                layer.params[key] = np.asarray(weights[index], dtype=np.float64)
                index += 1

    def forward(self, x, training=True):
        out = x
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def _collect_params_and_grads(self):
        for layer in self.layers:
            if getattr(layer, "trainable", True):
                for param, grad in layer.get_params_and_grads():
                    yield param, grad

    def predict(self, x, batch_size=None):
        if batch_size is None:
            return self.forward(x, training=False)
        outputs = []
        for i in range(0, len(x), batch_size):
            outputs.append(self.forward(x[i : i + batch_size], training=False))
        return np.concatenate(outputs, axis=0)

    def evaluate(self, x, y, batch_size=32):
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(x), batch_size):
            xb, yb = x[i : i + batch_size], y[i : i + batch_size]
            y_pred = self.forward(xb, training=False)
            total_loss += self.loss_fn(y_pred, yb)
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def fit(
        self,
        x,
        y,
        epochs=10,
        batch_size=32,
        validation_data=None,
        shuffle=True,
        verbose=True,
        callbacks=None,
    ):
        n = len(x)
        callback_list = list(callbacks or [])
        self.stop_training = False

        for callback in callback_list:
            if hasattr(callback, "set_model"):
                callback.set_model(self)
            if hasattr(callback, "on_train_begin"):
                callback.on_train_begin({})

        for epoch in range(1, epochs + 1):
            for callback in callback_list:
                if hasattr(callback, "on_epoch_begin"):
                    callback.on_epoch_begin(epoch - 1, {})

            if shuffle:
                idx = np.random.permutation(n)
                x, y = x[idx], y[idx]

            epoch_loss, n_batches = 0.0, 0
            for i in range(0, n, batch_size):
                xb, yb = x[i : i + batch_size], y[i : i + batch_size]
                y_pred = self.forward(xb, training=True)
                loss = self.loss_fn(y_pred, yb)
                grad = self.loss_fn.backward()
                self.backward(grad)
                self.optimizer.step(self._collect_params_and_grads())
                epoch_loss += loss
                n_batches += 1

            epoch_loss /= max(n_batches, 1)
            self.history["loss"].append(epoch_loss)
            log = f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f}"

            if validation_data is not None:
                val_x, val_y = validation_data
                val_loss = self.evaluate(val_x, val_y, batch_size=batch_size)
                self.history["val_loss"].append(val_loss)
                log += f" - val_loss: {val_loss:.4f}"

            logs = {"loss": epoch_loss}
            if validation_data is not None:
                logs["val_loss"] = val_loss

            for callback in callback_list:
                if hasattr(callback, "on_epoch_end"):
                    callback.on_epoch_end(epoch - 1, logs)

            if verbose:
                print(log)

            if self.stop_training:
                break

        for callback in callback_list:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end({"loss": self.history["loss"], "val_loss": self.history["val_loss"]})

        return self.history

    def summary(self):
        print(f"{'Layer':<20}{'Output Shape':<20}{'Param #':<10}")
        print("-" * 50)
        total_params = 0
        for layer in self.layers:
            n_params = sum(p.size for p in layer.params.values())
            total_params += n_params
            print(f"{layer.name:<20}{'?':<20}{n_params:<10}")
        print("-" * 50)
        print(f"Total trainable params: {total_params}")