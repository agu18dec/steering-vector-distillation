"""PreconditionedSGD + SignSGD.

PreconditionedSGD update: p -= lr * scale[p] * grad
    where scale[p] = 1/sqrt(v+eps) from a fixed Adam second-moment estimate.
    Isolates Adam's coordinate-wise scaling from momentum / running stats.

SignSGD update: p -= lr * sign(grad)
    Uniform per-coordinate step magnitude. Isolates step-magnitude uniformity
    from any running statistics.
"""

import torch
from torch.optim import Optimizer


class PreconditionedSGD(Optimizer):
    def __init__(self, params, lr, scales, param_names):
        if scales is None or param_names is None:
            raise ValueError("scales and param_names are required")
        defaults = dict(lr=lr)
        self._scales = scales
        self._param_names = param_names
        super().__init__(params, defaults)

        missing = []
        for group in self.param_groups:
            for p in group["params"]:
                name = self._param_names.get(id(p))
                if name is None or name not in self._scales:
                    missing.append(name)
        if missing:
            print(
                f"[PreconditionedSGD] WARNING: {len(missing)} trainable params without "
                "scales — they will use unscaled SGD"
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                name = self._param_names.get(id(p))
                if name is not None and name in self._scales:
                    scale = self._scales[name].to(p.device, dtype=p.dtype)
                    p.add_(p.grad * scale, alpha=-lr)
                else:
                    p.add_(p.grad, alpha=-lr)
        return loss


class SignSGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad.sign(), alpha=-lr)
        return loss
