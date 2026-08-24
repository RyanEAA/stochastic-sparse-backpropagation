"""Fast correctness smoke tests with no dataset download."""
import torch
import torch.nn as nn

from algorithms.ssb.registry import SSB_LAYERS
from algorithms.ssb.v0 import SparseLinearV0
from algorithms.ssb.v1 import SparseLinearV1
from algorithms.ssb.v2 import SparseLinearV2
from algorithms.ssb.v3 import SparseLinearV3
from algorithms.ssb.v3.function import SparseLinearFunctionV3
from algorithms.ssb.v1_block import BlockSparseLinearV1
from algorithms.ssb.v2_block import BlockSparseLinearV2
from algorithms.ssb.v3_block import BlockSparseLinearV3
from models import AVAILABLE_MODELS, build_model
from training.runtime import initialize_model_parameters


def assert_same_gradients(left, right, x, grad_output, seed=None):
    right.load_state_dict(left.state_dict())
    x_left = x.detach().clone().requires_grad_(True)
    x_right = x.detach().clone().requires_grad_(True)
    if seed is not None:
        torch.manual_seed(seed)
    y_left = left(x_left)
    if seed is not None:
        torch.manual_seed(seed)
    y_right = right(x_right)
    y_left.backward(grad_output)
    y_right.backward(grad_output)
    assert torch.allclose(y_left, y_right)
    assert torch.allclose(x_left.grad, x_right.grad)
    assert torch.allclose(left.weight.grad, right.weight.grad)
    assert torch.allclose(left.bias.grad, right.bias.grad)


def parameter_tensors(model):
    return [parameter.detach() for parameter in model.parameters() if parameter.ndim in (1, 2, 4)]


def assert_initialization_parity(dataset, architecture, left_name, right_name, keep_ratio=0.5):
    left = build_model(dataset, left_name, keep_ratio, architecture=architecture)
    right = build_model(dataset, right_name, keep_ratio, architecture=architecture)
    initialize_model_parameters(left, seed=77)
    initialize_model_parameters(right, seed=77)
    left_parameters = parameter_tensors(left)
    right_parameters = parameter_tensors(right)
    assert len(left_parameters) == len(right_parameters)
    for left_parameter, right_parameter in zip(left_parameters, right_parameters):
        assert torch.equal(left_parameter, right_parameter), (
            f"initialization mismatch: {dataset}/{architecture}/{left_name}/{right_name}"
        )


def test_v3_selected_forward_and_backward():
    torch.manual_seed(2)
    x = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    bias = torch.randn(5, requires_grad=True)
    mask = torch.tensor([True, False, True, False, False])
    output = SparseLinearFunctionV3.apply(x, weight, bias, mask)

    expected = torch.zeros(3, 5)
    expected[:, mask] = x.detach() @ weight.detach()[mask].t() + bias.detach()[mask]
    assert torch.allclose(output.detach(), expected)
    assert torch.count_nonzero(output[:, ~mask]) == 0

    output.sum().backward()
    assert torch.count_nonzero(weight.grad[~mask]) == 0
    assert torch.count_nonzero(bias.grad[~mask]) == 0


def test_block_mask_structure():
    layer = BlockSparseLinearV1(7, 23, keep_ratio=0.5, block_size=4)
    torch.manual_seed(123)
    mask = layer._sample_mask()
    for start in range(0, mask.numel(), layer.block_size):
        block = mask[start:start + layer.block_size]
        assert bool((block == block[0]).all()), "block mask is not contiguous/aligned"


def test_v3_block_selected_forward_and_backward():
    layer = BlockSparseLinearV3(4, 8, keep_ratio=0.5, block_size=2)
    layer.train()
    with torch.no_grad():
        layer.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10.0)
        layer.bias.copy_(torch.arange(8, dtype=torch.float32) / 10.0)

    x = torch.randn(3, 4, requires_grad=True)
    forced_mask = torch.tensor([True, True, False, False, True, True, False, False])
    layer._sample_mask = lambda: forced_mask.to(layer.weight.device)

    output = layer(x)
    assert torch.count_nonzero(output[:, ~forced_mask]) == 0
    expected_active = x.detach() @ layer.weight.detach()[forced_mask].t() + layer.bias.detach()[forced_mask]
    assert torch.allclose(output.detach()[:, forced_mask], expected_active)

    output.sum().backward()
    assert torch.count_nonzero(layer.weight.grad[~forced_mask]) == 0
    assert torch.count_nonzero(layer.bias.grad[~forced_mask]) == 0



def one_step(dataset, architecture, model_name, num_classes):
    shape = (4, 1, 28, 28) if dataset in {"mnist", "fashion_mnist", "kmnist"} else (4, 3, 32, 32)
    x = torch.randn(*shape)
    y = torch.randint(0, num_classes, (shape[0],))
    model = build_model(
        dataset,
        model_name,
        keep_ratio=0.5,
        architecture=architecture,
        block_size=16,
    )
    initialize_model_parameters(model, seed=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    optimizer.step()
    assert output.shape == (shape[0], num_classes)
    assert torch.isfinite(loss)
    print(f"PASS {dataset} {architecture} {model_name}: loss={loss.item():.4f}")


def main():
    expected = ("ssb-v0", "ssb-v1", "ssb-v2", "ssb-v3", "ssb-v1-block", "ssb-v2-block", "ssb-v3-block")
    assert tuple(SSB_LAYERS) == expected
    assert all(name in AVAILABLE_MODELS for name in expected)

    torch.manual_seed(10)
    x = torch.randn(4, 7)
    grad_output = torch.randn(4, 5)
    assert_same_gradients(SparseLinearV0(7, 5, 0.6), SparseLinearV1(7, 5, 0.6), x, grad_output, seed=123)
    assert_same_gradients(SparseLinearV1(7, 5, 1.0), SparseLinearV2(7, 5, 1.0), x, grad_output)
    assert_same_gradients(SparseLinearV1(7, 5, 1.0), SparseLinearV3(7, 5, 1.0), x, grad_output)
    assert_same_gradients(SparseLinearV1(7, 5, 1.0), BlockSparseLinearV1(7, 5, 1.0, block_size=2), x, grad_output)
    assert_same_gradients(SparseLinearV3(7, 5, 1.0), BlockSparseLinearV3(7, 5, 1.0, block_size=2), x, grad_output)

    test_v3_selected_forward_and_backward()
    test_block_mask_structure()
    test_v3_block_selected_forward_and_backward()

    for architecture in ("mlp", "cnn"):
        assert_initialization_parity("cifar10", architecture, "dense", "ssb-v1")
        assert_initialization_parity("cifar10", architecture, "dense", "ssb-v3")
        one_step("cifar10", architecture, "dense", 10)
        one_step("cifar10", architecture, "ssb-v3", 10)
        one_step("cifar10", architecture, "ssb-v1-block", 10)
        one_step("cifar10", architecture, "ssb-v2-block", 10)
        one_step("cifar10", architecture, "ssb-v3-block", 10)

    print("PASS registry, V0/V1/V2 compatibility, V3/V3-block semantics, block structure, CNN/MLP wiring, and initialization parity")


if __name__ == "__main__":
    main()
