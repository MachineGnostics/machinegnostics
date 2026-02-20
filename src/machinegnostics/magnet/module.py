import numpy as np
import pickle

from .layers import BaseLayer, ParamLayer


class Module(object):
    def __init__(self):
        self._ordered_layers = []
        self.params = {}

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, loss):
        if not hasattr(loss, "delta"):
            raise TypeError("loss must provide a 'delta' attribute")
        # find net order
        layers = []
        for name, v in self.__dict__.items():
            if not isinstance(v, BaseLayer):
                continue
            layer = v
            layer.name = name
            layers.append((layer.order, layer))
        self._ordered_layers = [l[1] for l in sorted(layers, key=lambda x: x[0])]

        # back propagate through this order
        last_layer = self._ordered_layers[-1]
        last_layer.data_vars["out"].set_error(loss.delta)
        for layer in self._ordered_layers[::-1]:
            grads = layer.backward()
            if isinstance(layer, ParamLayer):
                for k in layer.param_vars.keys():
                    self.params[layer.name]["grads"][k][:] = grads[k]

    def save(self, path):
        vars_map = {name: p["vars"] for name, p in self.params.items()}
        with open(path, "wb") as f:
            pickle.dump(vars_map, f)

    def restore(self, path):
        with open(path, "rb") as f:
            params = pickle.load(f)
        for name, param in params.items():
            if name not in self.params:
                continue
            for p_name in self.params[name]["vars"].keys():
                self.params[name]["vars"][p_name][:] = param[p_name]

    def sequential(self, *layers):
        assert isinstance(layers, (list, tuple))
        for i, l in enumerate(layers):
            self.__setattr__("layer_%i" % i, l)
        return SeqLayers(layers)

    def __call__(self, *args):
        return self.forward(*args)

    def __setattr__(self, key, value):
        if isinstance(value, ParamLayer):
            layer = value
            self.params[key] = {
                "vars": layer.param_vars,
                "grads": {k: np.empty_like(layer.param_vars[k]) for k in layer.param_vars.keys()}
            }
        object.__setattr__(self, key, value)


class SeqLayers:
    def __init__(self, layers):
        assert isinstance(layers, (list, tuple))
        for l in layers:
            assert isinstance(l, BaseLayer)
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)