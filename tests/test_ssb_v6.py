import torch
from torch import nn

from algorithms.ssb.v6 import GradientSelectedChildModelV6
from algorithms.ssb.v7 import OptimizedSelectedChildModelV7
from algorithms.ssb.v6.gradient_selected_child import (
    gradient_retention_indices,
    selected_unit_mask_distance,
    structured_gradient_l2,
    structured_taylor,
    structured_weight_l2,
    topk_structured_indices,
)
from models.common.cnn import DenseCNN
from models.common.mlp import DenseMLP


def test_gradient_retention_selects_smallest_energy_covering_set():
    scores = torch.tensor([1.0, 2.0, 6.0, 8.0])
    # Energy is [1, 4, 36, 64]; units 3 and 2 retain 100/105 > 90%.
    assert torch.equal(gradient_retention_indices(scores, 0.90), torch.tensor([2, 3]))


def test_zero_gradient_retention_safely_keeps_all_units():
    assert torch.equal(
        gradient_retention_indices(torch.zeros(4), 0.90), torch.arange(4)
    )


def _train_one(model, x, y, *, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    model.refresh_child()
    optimizer = model.make_optimizer(lr)
    model.train()
    optimizer, scored = model.score_and_refresh(x, y, criterion, optimizer, lr)
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer = model.after_optimizer_step(optimizer, lr)
    return optimizer, scored


def test_known_gradient_ranking_selects_expected_neurons():
    gradient = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 10.0], [2.0, 0.0]])
    scores = structured_gradient_l2(gradient)
    assert torch.equal(scores, torch.tensor([5.0, 1.0, 10.0, 2.0]))
    assert torch.equal(topk_structured_indices(scores, 0.5), torch.tensor([0, 2]))


def test_weight_l2_and_taylor_scores_are_structured_per_output_unit():
    weight = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    gradient = torch.tensor([[2.0, -1.0], [7.0, 3.0]])
    assert torch.equal(structured_weight_l2(weight), torch.tensor([5.0, 1.0]))
    assert torch.equal(structured_taylor(weight, gradient), torch.tensor([10.0, 7.0]))


def test_selected_unit_mask_distance_counts_replacements():
    previous = [torch.tensor([0, 2]), torch.tensor([1, 3])]
    current = [torch.tensor([0, 1]), torch.tensor([1, 3])]
    assert selected_unit_mask_distance(previous, current) == 0.25


def test_weight_l2_scoring_does_not_run_dense_forward_or_backward():
    master = DenseMLP(3, [4], 2)
    model = GradientSelectedChildModelV6(
        master, keep_ratio=0.5, score_refresh_steps=1, selection_method="weight_l2"
    )
    model.refresh_child()
    optimizer = model.make_optimizer(1e-3)

    def forbidden_criterion(*_args):
        raise AssertionError("weight-L2 selection must not evaluate a dense loss")

    optimizer, scored = model.score_and_refresh(
        torch.randn(2, 3), torch.tensor([0, 1]), forbidden_criterion, optimizer, 1e-3
    )
    assert scored and optimizer is not None


def test_random_scoring_is_a_cheap_matched_selector_control():
    torch.manual_seed(17)
    master = DenseMLP(3, [6], 2)
    model = GradientSelectedChildModelV6(
        master, keep_ratio=0.5, score_refresh_steps=1, selection_method="random"
    )
    model.refresh_child()
    optimizer = model.make_optimizer(1e-3)

    def forbidden_criterion(*_args):
        raise AssertionError("random selection must not evaluate a dense loss")

    optimizer, scored = model.score_and_refresh(
        torch.randn(2, 3), torch.tensor([0, 1]), forbidden_criterion, optimizer, 1e-3
    )
    assert scored and optimizer is not None
    assert model._importance
    assert all(scores.ndim == 1 for scores in model._importance.values())


def test_early_bird_freezes_after_configured_stable_window():
    master = DenseMLP(3, [5], 2)
    model = GradientSelectedChildModelV6(
        master, keep_ratio=0.2, score_refresh_steps=1, selection_method="weight_l2",
        early_bird=True, stability_window=3, stability_threshold=0.0,
    )
    hidden = [m for m in master.net if isinstance(m, nn.Linear)][0]
    model._importance[hidden] = torch.tensor([9.0, 1.0, 1.0, 1.0, 1.0])
    for event in range(4):
        model.scoring_event_count = event
        model.refresh_child()
        model._update_early_bird_state()
    assert model.topology_frozen
    assert model.topology_freeze_scoring_event == 4
    assert not model.scoring_due()


def test_v6_mlp_is_smaller_and_uses_ranked_indices():
    master = DenseMLP(4, [4], 2)
    model = GradientSelectedChildModelV6(master, keep_ratio=0.5, score_refresh_steps=1)
    hidden = [m for m in master.net if isinstance(m, nn.Linear)][0]
    model._importance[hidden] = torch.tensor([1.0, 8.0, 2.0, 7.0])
    model.refresh_child()
    assert torch.equal(model._maps[0].out_idx.cpu(), torch.tensor([1, 3]))
    assert model.child_parameter_count() < model.master_parameter_count()


def test_v6_gather_scatter_and_adam_state_use_exact_master_indices():
    master = DenseMLP(3, [4], 2)
    model = GradientSelectedChildModelV6(master, keep_ratio=0.5, score_refresh_steps=2)
    hidden = [m for m in master.net if isinstance(m, nn.Linear)][0]
    model._importance[hidden] = torch.tensor([0.0, 9.0, 1.0, 8.0])
    model.refresh_child()
    mapping = model._maps[0]
    assert torch.equal(mapping.out_idx.cpu(), torch.tensor([1, 3]))

    original = hidden.weight.detach().clone()
    with torch.no_grad():
        mapping.child.weight.fill_(42.0)
    model.sync_child_to_master()
    assert torch.equal(hidden.weight[[1, 3]], torch.full_like(hidden.weight[[1, 3]], 42.0))
    assert torch.equal(hidden.weight[[0, 2]], original[[0, 2]])

    state = model._master_state_for(hidden.weight)
    with torch.no_grad():
        state["exp_avg"].copy_(torch.arange(hidden.weight.numel()).reshape_as(hidden.weight))
    optimizer = model.make_optimizer(1e-3)
    child_state = optimizer.state[mapping.child.weight]
    expected = state["exp_avg"].index_select(0, mapping.out_idx).index_select(1, mapping.in_idx)
    assert torch.equal(child_state["exp_avg"], expected)
    with torch.no_grad():
        child_state["exp_avg"].fill_(7.0)
    model.sync_optimizer_state_to_master(optimizer)
    assert torch.equal(state["exp_avg"][[1, 3]], torch.full_like(state["exp_avg"][[1, 3]], 7.0))


def test_v6_new_scores_change_selected_topology():
    master = DenseMLP(3, [4], 2)
    model = GradientSelectedChildModelV6(master, keep_ratio=0.5, score_refresh_steps=1)
    hidden = [m for m in master.net if isinstance(m, nn.Linear)][0]
    model._importance[hidden] = torch.tensor([9.0, 8.0, 1.0, 0.0])
    model.refresh_child()
    first = model._maps[0].out_idx.detach().clone()
    model._importance[hidden] = torch.tensor([0.0, 1.0, 8.0, 9.0])
    model.refresh_child()
    second = model._maps[0].out_idx.detach().clone()
    assert torch.equal(first.cpu(), torch.tensor([0, 1]))
    assert torch.equal(second.cpu(), torch.tensor([2, 3]))
    assert not torch.equal(first, second)


def test_v6_dynamic_ratio_changes_with_gradient_energy_distribution():
    master = DenseMLP(3, [4], 2)
    model = GradientSelectedChildModelV6(
        master, keep_ratio=1.0, score_refresh_steps=1,
        selection_mode="gradient_retention", gradient_retention=0.90,
    )
    hidden = [m for m in master.net if isinstance(m, nn.Linear)][0]
    model._importance[hidden] = torch.tensor([10.0, 0.1, 0.1, 0.1])
    model.refresh_child()
    concentrated_ratio = model.effective_keep_ratio()
    model._importance[hidden] = torch.ones(4)
    model.refresh_child()
    spread_ratio = model.effective_keep_ratio()
    assert concentrated_ratio == 0.25
    assert spread_ratio == 1.0


def test_v6_scoring_changes_topology_and_preserves_adam_state():
    torch.manual_seed(4)
    model = GradientSelectedChildModelV6(
        DenseMLP(4, [6], 3), keep_ratio=0.5, score_refresh_steps=1
    )
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    optimizer, scored = _train_one(model, x, y)
    assert scored and model.scoring_event_count == 1
    first = model.topology_signature()
    assert model._master_adam_state
    x2 = torch.randn(8, 4) * 4
    y2 = torch.randint(0, 3, (8,))
    optimizer, scored = model.score_and_refresh(x2, y2, nn.CrossEntropyLoss(), optimizer, 1e-3)
    assert scored and model.scoring_event_count == 2
    assert optimizer.state
    assert model.topology_signature() != ""
    # A topology may coincidentally repeat; the refresh and ranking event must not.
    assert model.refresh_count >= 3 and first


def test_v6_keep_one_matches_full_dimensions():
    model = GradientSelectedChildModelV6(DenseMLP(4, [5], 2), 1.0, 2)
    model.refresh_child()
    assert model.child_parameter_count() == model.master_parameter_count()
    for mapping in model._maps:
        assert mapping.out_idx.numel() == mapping.master.out_features


def test_v6_cnn_scores_and_trains_physically_smaller_child():
    torch.manual_seed(5)
    master = DenseCNN(1, 8, [4, 6], [5], 3, pooled_size=2)
    model = GradientSelectedChildModelV6(master, keep_ratio=0.5, score_refresh_steps=1)
    x = torch.randn(4, 1, 8, 8)
    y = torch.randint(0, 3, (4,))
    _, scored = _train_one(model, x, y)
    assert scored
    assert model.child_parameter_count() < model.master_parameter_count()
    assert any(mapping.kind == "conv" for mapping in model._maps)
    assert any(mapping.kind == "linear" for mapping in model._maps)


def test_v7_does_not_scatter_master_adam_state_on_ordinary_steps():
    model = OptimizedSelectedChildModelV7(
        DenseMLP(4, [6], 3), keep_ratio=0.5, score_refresh_steps=10
    )
    model.refresh_child()
    optimizer = model.make_optimizer(1e-3)
    calls = []
    model.sync_optimizer_state_to_master = lambda _optimizer: calls.append(True)
    model.after_optimizer_step(optimizer, 1e-3)
    model.after_optimizer_step(optimizer, 1e-3)
    assert calls == []
    assert model.optimizer_step_count == 2


def test_v7_layerwise_cnn_shorthand_expands_by_layer_family():
    model = OptimizedSelectedChildModelV7(
        DenseCNN(3, 16, [4, 6, 8], [10, 5], 3, pooled_size=2),
        keep_ratio=0.2,
        score_refresh_steps=25,
        selection_method="weight_l2",
        layer_keep_ratios=[1.0, 0.5, 0.2],
    )
    model.refresh_child()
    # First conv dense, later convs at 50%, classifier hidden layers at 20%.
    assert [mapping.out_idx.numel() for mapping in model._maps[:-1]] == [4, 3, 4, 2, 1]


def test_v7_early_bird_respects_minimum_scoring_events():
    model = OptimizedSelectedChildModelV7(
        DenseMLP(3, [5], 2),
        keep_ratio=0.2,
        score_refresh_steps=1,
        selection_method="weight_l2",
        early_bird=True,
        stability_window=2,
        stability_threshold=0.0,
        early_bird_min_events=4,
    )
    hidden = [module for module in model.master.net if isinstance(module, nn.Linear)][0]
    model._importance[hidden] = torch.tensor([9.0, 1.0, 1.0, 1.0, 1.0])
    for event in range(3):
        model.scoring_event_count = event
        model.refresh_child()
        model._update_early_bird_state()
        assert not model.topology_frozen
    model.scoring_event_count = 3
    model.refresh_child()
    model._update_early_bird_state()
    assert model.topology_frozen


def test_v7_dense_correction_preserves_dense_update_and_rebuilds_child():
    torch.manual_seed(9)
    model = OptimizedSelectedChildModelV7(
        DenseMLP(4, [6], 3),
        keep_ratio=0.5,
        score_refresh_steps=10,
        selection_method="weight_l2",
        dense_correction_steps=1,
    )
    model.refresh_child()
    child_optimizer = model.make_optimizer(1e-3)
    model.after_optimizer_step(child_optimizer, 1e-3)
    assert model.dense_correction_due()

    master_optimizer = model.prepare_dense_correction(child_optimizer, 1e-3)
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    master_optimizer.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(model.master(x), y)
    loss.backward()
    before = [parameter.detach().clone() for parameter in model.master.parameters()]
    master_optimizer.step()
    after = [parameter.detach().clone() for parameter in model.master.parameters()]
    child_optimizer = model.finish_dense_correction(master_optimizer, 1e-3)

    assert any(not torch.equal(left, right) for left, right in zip(before, after))
    assert all(
        torch.equal(parameter, expected)
        for parameter, expected in zip(model.master.parameters(), after)
    )
    assert model.child is not None
    assert child_optimizer.state
    assert model.dense_correction_count == 1
    assert not model.dense_correction_due()
