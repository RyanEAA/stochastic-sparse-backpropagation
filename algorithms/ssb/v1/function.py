import torch
from torch.autograd import Function


class SparseLinearFunctionV1(Function):
    """Clean SSB backward.

    Same gradient rule as V0, but without active-fraction/gradient-density
    instrumentation in the backward hot path.
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
        else:
            grad_x = torch.zeros_like(x)

        return grad_x, grad_weight, grad_bias, None
