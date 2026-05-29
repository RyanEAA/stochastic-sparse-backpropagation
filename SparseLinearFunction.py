import torch
from torch.autograd import Function

class SparseLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, active_mask):

        # save tensors for backward
        ctx.save_for_backward(
            x,
            weight,
            bias,
            active_mask
        )

        # standard dense forward
        return x @ weight.t() + bias

    @staticmethod
    def backward(ctx, grad_output):

        x, weight, bias, active_mask = ctx.saved_tensors

        masked_grad_output = grad_output.clone()
        masked_grad_output[:, ~active_mask] = 0

        grad_x = masked_grad_output @ weight

        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)

        active_idx = active_mask.nonzero(as_tuple=True)[0]

        if active_idx.numel() > 0:

            #scale = 1.0 / active_mask.float().mean() # breaks around 59%

            #scale = 1.0 / torch.sqrt(active_mask.float().mean()) # breaks around 60%
            scale = 1.0 / torch.sqrt(active_mask.float().mean() + 1e-8) # breaks around 50% but stable, 1e-8 is added to prevent division by zero when keep_ratio is very small
            
            #scale = 1.0 / torch.log1p(active_mask.float().mean()) # breaks around 60%
            #scale = 1.0 / torch.sqrt(active_mask.float().mean() + 1e-8) # breaks around 60%
            
            #scale = 1.0 / torch.sqrt(active_mask.float().mean())/active_mask().mean() # breaks around 59%
            #scale = 1.0 / torch.sqrt(active_mask.float().mean()+1e-8)/active_mask().mean() # breaks around 60%

            #scale = 1.0 / torch.exp(torch.log(active_mask.float().mean()) + 1e-8) # breaks around 59%

            #scale = 1.0 / torch.exp(active_idx.float().mean()/active_mask.float().mean() + 1e-8) # breaks around 59%
            #scale = 1.0 / torch.exp(active_idx.float().mean()) # breaks around 55%
            
            #scale = torch.log(active_idx.float().mean() + 1e-8) # breaks around 60%

            grad_weight[active_idx] = (
                masked_grad_output[:, active_idx].t() @ x
            ) * scale

            grad_bias[active_idx] = (
                masked_grad_output[:, active_idx].sum(dim=0)
            ) * scale

        return grad_x, grad_weight, grad_bias, None