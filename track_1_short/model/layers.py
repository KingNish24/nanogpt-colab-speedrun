"""Small building blocks shared across the model."""
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinearT(nn.Module):
    """
    Linear layer with transposed weight storage (in_features, out_features) which
    addresses the slow kernel that was used for gradient accumulation. @chrisjmccormick

    Its forward is the plain fp16 path (both training and validation on T4); the fp8 scales x_s,
    w_s and grad_s it once carried for the fused fp8 loss are kept only for API compatibility.
    """
    def __init__(self, in_features: int, out_features: int, x_s: float, w_s: float, grad_s: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.float16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.weight) # @Grad62304977 and others

    def forward(self, x: Tensor):
        return x @ self.weight.type_as(x)


def next_multiple_of_n(v: float | int, *, n: int):
    return math.ceil(v / n) * n
