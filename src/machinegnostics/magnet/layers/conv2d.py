"""
Minimal Conv2D layer (NHWC) for CNNs.
"""
import numpy as np
from .base import BaseLayer

class Conv2D(BaseLayer):
    def __init__(self, filters: int, kernel_size: tuple = (3,3), stride: int = 1, padding: str = 'same', activation: str = None, neuron_type: str = 'E'):
        super().__init__(name=f"Conv2D({filters},{kernel_size})", neuron_type=neuron_type)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        from machinegnostics.magnet.activations import get_activation
        if activation is None:
            activation = 'relu' if neuron_type == 'E' else 'quadratic'
        self.activation_name = activation
        self.activation, self.activation_grad = get_activation(activation)

    def build(self, input_shape):
        # input_shape: (batch, H, W, C)
        _, H, W, C = input_shape
        kh, kw = self.kernel_size
        scale = np.sqrt(2.0 / (kh * kw * C)) if self.neuron_type == 'E' else np.sqrt(1.0 / (kh * kw * C))
        W = np.random.randn(kh, kw, C, self.filters) * scale
        b = np.zeros((1, 1, 1, self.filters))
        self.params = {'W': W, 'b': b}
        self.built = True
        self.input_shape = input_shape
        self.output_shape = input_shape  # will adjust in forward

    def _pad(self, x, pad_h, pad_w):
        return np.pad(x, ((0,0),(pad_h,pad_h),(pad_w,pad_w),(0,0)), mode='constant')

    def forward(self, x):
        self.x = x
        batch, H, W, C = x.shape
        kh, kw = self.kernel_size
        sh = sw = self.stride
        if self.padding == 'same':
            out_h = int(np.ceil(H / sh))
            out_w = int(np.ceil(W / sw))
            pad_h = max((out_h - 1) * sh + kh - H, 0)
            pad_w = max((out_w - 1) * sw + kw - W, 0)
            pad_top = pad_h // 2
            pad_left = pad_w // 2
        else:
            pad_top = pad_left = 0
            out_h = (H - kh) // sh + 1
            out_w = (W - kw) // sw + 1
        x_p = self._pad(x, pad_top, pad_left)
        W = self.params['W']
        b = self.params['b']
        out = np.zeros((batch, out_h, out_w, self.filters))
        for i in range(out_h):
            for j in range(out_w):
                hs = i * sh
                ws = j * sw
                x_slice = x_p[:, hs:hs+kh, ws:ws+kw, :]  # (batch, kh, kw, C)
                # (batch, filters)
                out[:, i, j, :] = np.tensordot(x_slice, W, axes=([1,2,3],[0,1,2])) + b.reshape(1,1,1,-1)
        self.z = out
        self.output_shape = out.shape
        return self.activation(out)

    def backward(self, grad_out):
        # naive backward (slow), suitable for small examples
        batch, H, W, C = self.x.shape
        kh, kw = self.kernel_size
        sh = sw = self.stride
        if self.padding == 'same':
            out_h = int(np.ceil(H / sh))
            out_w = int(np.ceil(W / sw))
            pad_h = max((out_h - 1) * sh + kh - H, 0)
            pad_w = max((out_w - 1) * sw + kw - W, 0)
            pad_top = pad_h // 2
            pad_left = pad_w // 2
        else:
            pad_top = pad_left = 0
            out_h = (H - kh) // sh + 1
            out_w = (W - kw) // sw + 1
        x_p = self._pad(self.x, pad_top, pad_left)
        dz = grad_out * self.activation_grad(self.z)
        dW = np.zeros_like(self.params['W'])
        db = np.sum(dz, axis=(0,1,2), keepdims=True)
        dx_p = np.zeros_like(x_p)
        for i in range(out_h):
            for j in range(out_w):
                hs = i * sh
                ws = j * sw
                x_slice = x_p[:, hs:hs+kh, ws:ws+kw, :]
                for f in range(self.filters):
                    dW[:,:, :, f] += np.sum(x_slice * dz[:, i, j, f][:, None, None, None], axis=0)
                # accumulate gradients to input
                dx_p[:, hs:hs+kh, ws:ws+kw, :] += np.tensordot(dz[:, i, j, :], self.params['W'], axes=([1],[3]))
        # remove padding
        dx = dx_p[:, pad_top:pad_top+H, pad_left:pad_left+W, :]
        self.grads = {'W': dW, 'b': db}
        return dx
