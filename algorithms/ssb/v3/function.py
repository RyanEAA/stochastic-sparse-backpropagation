import torch
from torch.autograd import Function


class SparseLinearFunctionV3(Function):
    """SSB with the same stochastic neuron selection in forward and backward.

    During training, only active output neurons are computed. The reduced result
    is scattered into a full-width output tensor so surrounding PyTorch modules
    keep the normal dense shape contract. Inactive outputs are exactly zero and
    do not participate in the backward computation.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, active_mask):
        active_idx = active_mask.nonzero(as_tuple=True)[0]
        ctx.save_for_backward(x, weight, active_idx)
        ctx.has_bias = bias is not None

        output = x.new_zeros((*x.shape[:-1], weight.size(0)))
        if active_idx.numel() == 0:
            return output

        active_output = x @ weight[active_idx].t()
        if bias is not None:
            active_output = active_output + bias[active_idx]
        output[..., active_idx] = active_output
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, active_idx = ctx.saved_tensors
        grad_weight = torch.zeros_like(weight)
        grad_bias = (
            torch.zeros(weight.size(0), device=weight.device, dtype=grad_output.dtype)
            if ctx.has_bias
            else None
        )

        if active_idx.numel() == 0:
            grad_x = torch.zeros_like(x)
        else:
            grad_output_active = grad_output[..., active_idx]
            weight_active = weight[active_idx]
            grad_x = grad_output_active @ weight_active

            x_2d = x.reshape(-1, x.shape[-1])
            grad_2d = grad_output_active.reshape(-1, grad_output_active.shape[-1])
            grad_weight[active_idx] = grad_2d.t() @ x_2d
            if grad_bias is not None:
                grad_bias[active_idx] = grad_2d.sum(dim=0)

        return grad_x, grad_weight, grad_bias, None
