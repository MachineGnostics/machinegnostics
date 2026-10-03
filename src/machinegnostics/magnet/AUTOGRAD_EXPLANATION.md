# MAGNET loss backpropagation in the new implementation

The new MAGNET losses do **not** implement manual `backward()` methods because
they rely on **PyTorch autograd** through the `machinegnostics.magnet.core.Tensor`
facade.

## How gradients flow

1. MAGNET loss classes compute with `torch.Tensor` values internally.
2. They return results with `Tensor.from_torch(...)`.
3. That returned MAGNET `Tensor` still wraps the original PyTorch computation graph.
4. Calling `loss.backward()` triggers PyTorch's automatic differentiation.
5. Gradients are written back to trainable tensors such as dense weights, `Fi.S`, and `Fi.z0`.

## Why there is no explicit `backward()` in the new losses

The old loss implementations manually derived gradients. The refactored loss files
instead build the loss directly from differentiable PyTorch operations such as
`torch.mean`, `torch.cosh`, `torch.tanh`, and `torch.log`. PyTorch already knows
how to differentiate these operations efficiently and reliably, so custom backward
code is unnecessary.

This is the standard PyTorch pattern and is easier to maintain because the forward
formula and the gradient logic stay consistent automatically.

## Example

```python
import numpy as np
from machinegnostics.magnet import Dense, Fi, FidelityLoss, Sequential, SGD

model = Sequential([Dense(2, 1), Fi()])
model.compile(loss=FidelityLoss(), optimizer=SGD(lr=0.01))

x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
y = np.array([[1.0], [1.0]], dtype=float)

prediction = model(x)
loss = model.loss_fn(prediction, y)
loss.backward()  # PyTorch autograd computes all required gradients
```

## Key takeaway

If a MAGNET loss returns a `Tensor`, gradient computation is automatic as long as
the forward pass was built from differentiable torch operations.

## Reference

See the PyTorch autograd documentation:
<https://pytorch.org/docs/stable/autograd.html>
