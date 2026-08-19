import torch
from torch.autograd import Function

from .metrics import add


class SparseLinearFunctionV0(Function):
    """Original SSB backward with historical instrumentation.

    Forward is dense. During backward, only randomly selected output neurons
    contribute to grad_input, grad_weight, and grad_bias. Gradients are not
    rescaled by keep_ratio.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, active_mask):
        ctx.save_for_backward(x, weight, bias, active_mask)
        return x @ weight.t() + bias

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, active_mask = ctx.saved_tensors
        active_idx = active_mask.nonzero(as_tuple=True)[0]

        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        if active_idx.numel() > 0:
            grad_output_active = grad_output[:, active_idx]
            weight_active = weight[active_idx]

            grad_x = grad_output_active @ weight_active
            grad_weight[active_idx] = grad_output_active.t() @ x

            if bias is not None:
                grad_bias[active_idx] = grad_output_active.sum(dim=0)

            # Historical instrumentation retained only in V0.
            try:
                active_frac = float(active_idx.numel()) / float(weight.size(0))
                active_grad = grad_weight[active_idx]
                if active_grad.numel() > 0:
                    grad_density = float((active_grad.abs() > 0).sum().item()) / float(active_grad.numel())
                else:
                    grad_density = 0.0
            except Exception:
                active_frac = 0.0
                grad_density = 0.0

            add("sparse_active_frac", active_frac)
            add("sparse_grad_density", grad_density)
        else:
            grad_x = torch.zeros_like(x)

        return grad_x, grad_weight, grad_bias, None
