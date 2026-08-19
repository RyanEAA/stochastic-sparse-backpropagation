import torch
from torch.autograd import Function


class SparseLinearFunctionV2(Function):
    """Inverse-probability-scaled SSB backward.

    Active gradients are multiplied by 1 / keep_ratio so that the stochastic
    gradient estimator matches the dense gradient in expectation.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, active_mask, keep_ratio):
        ctx.save_for_backward(x, weight, active_mask)
        ctx.keep_ratio = keep_ratio
        ctx.has_bias = bias is not None
        return x @ weight.t() + bias

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, active_mask = ctx.saved_tensors
        keep_ratio = ctx.keep_ratio
        active_idx = active_mask.nonzero(as_tuple=True)[0]

        grad_weight = torch.zeros_like(weight)
        grad_bias = (
            torch.zeros(
                weight.size(0), device=weight.device, dtype=grad_output.dtype
            )
            if ctx.has_bias
            else None
        )

        if active_idx.numel() == 0:
            grad_x = torch.zeros_like(x)
        else:
            grad_output_active = grad_output[:, active_idx]
            weight_active = weight[active_idx]
            scale = 1.0 / keep_ratio

            grad_x = (grad_output_active @ weight_active) * scale
            grad_weight[active_idx] = (grad_output_active.t() @ x) * scale

            if grad_bias is not None:
                grad_bias[active_idx] = grad_output_active.sum(dim=0) * scale

        return grad_x, grad_weight, grad_bias, None, None
