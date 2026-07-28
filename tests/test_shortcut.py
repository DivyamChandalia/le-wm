import math

import pytest
import torch

from jepa import ShortcutJEPA
from module import ShortcutFlowHead, ARPredictor, Embedder, MLP
from shortcut import (
    is_power_of_two,
    step_idx_from_nfe,
    sample_finest_flow_batch,
    flow_xpred_loss,
    sample_shortcut_grid,
    shortcut_bootstrap_loss,
    make_inference_schedule,
)


class MockEncoder(torch.nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.config = type("obj", (object,), {"hidden_size": dim})()
        self.register_buffer("_fixed_emb", torch.randn(256, 1, dim))

    def forward(self, pixels, interpolate_pos_encoding=True):
        B = pixels.size(0)
        idx = torch.arange(B) % self._fixed_emb.size(0)
        class MockOutput:
            last_hidden_state = self._fixed_emb[idx]
        return MockOutput()


def make_info_dict(B, S, H=3, T=6, act_dim=10, img_size=224):
    return {
        "pixels": torch.randn(B, H, 3, img_size, img_size),
        "goal": torch.randn(B, 1, 3, img_size, img_size),
        "action": torch.randn(B, H, act_dim),
    }, torch.randn(B, S, T, act_dim)


def test_is_power_of_two():
    assert is_power_of_two(1) is True
    assert is_power_of_two(2) is True
    assert is_power_of_two(4) is True
    assert is_power_of_two(8) is True
    assert is_power_of_two(3) is False
    assert is_power_of_two(0) is False
    assert is_power_of_two(-1) is False


def test_step_idx_from_nfe():
    assert step_idx_from_nfe(1) == 0
    assert step_idx_from_nfe(2) == 1
    assert step_idx_from_nfe(4) == 2
    assert step_idx_from_nfe(8) == 3


def test_sample_finest_flow_batch_shapes():
    B, T, D = 2, 3, 192
    target = torch.randn(B, T, D)
    k_max = 8
    sample = sample_finest_flow_batch(target, k_max)
    assert sample["x_t"].shape == (B, T, D)
    assert sample["noise"].shape == (B, T, D)
    assert sample["signal_idx"].shape == (B, T)
    assert sample["step_idx"].shape == (B, T)
    assert sample["tau"].shape == (B, T)
    assert sample["weight"].shape == (B, T)


def test_flow_xpred_loss_finite():
    pred = torch.randn(4, 3, 192, requires_grad=True)
    true = torch.randn(4, 3, 192)
    weight = torch.rand(4, 3)
    loss = flow_xpred_loss(pred, true, weight)
    assert loss.isfinite()
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.isfinite().all()


def test_shortcut_grid():
    B, T = 2, 3
    k_max = 8
    device = "cpu"
    grid = sample_shortcut_grid((B, T), k_max, device)
    assert grid["nfe"].shape == (B, T)
    assert grid["d"].shape == (B, T)
    assert grid["step_idx"].shape == (B, T)
    assert grid["signal_idx"].shape == (B, T)
    assert grid["tau"].shape == (B, T)
    assert grid["tau_plus_d"].shape == (B, T)
    assert grid["valid"].shape == (B, T)


def test_shortcut_grid_kmax_16_includes_nfe_8():
    B, T = 128, 10
    for k_max in [4, 8, 16]:
        device = "cpu"
        torch.manual_seed(42)
        grid = sample_shortcut_grid((B, T), k_max, device)
        unique_nfe = sorted(grid["nfe"].unique().tolist())
        max_exp = int(math.log2(k_max))
        expected = [2**e for e in range(max_exp)]
        assert unique_nfe == expected, f"k_max={k_max}: expected {expected}, got {unique_nfe}"


def test_shortcut_bootstrap_loss_finite():
    B, T, D = 2, 3, 192
    k_max = 8
    head = ShortcutFlowHead(dim=D, hidden_dim=512, k_max=k_max)
    x_t = torch.randn(B, T, D)
    context = torch.randn(B, T, D)
    signal_idx = torch.randint(0, 8, (B, T))
    step_idx = torch.zeros(B, T, dtype=torch.long)
    tau = torch.full((B, T), 0.25)
    d = torch.full((B, T), 0.25)
    loss = shortcut_bootstrap_loss(head, x_t, context, signal_idx, step_idx, tau, d, k_max)
    assert loss.isfinite()
    loss.backward()
    assert head.noisy_proj.weight.grad is not None


def test_make_inference_schedule():
    k_max = 8
    for nfe in [1, 2, 4, 8]:
        schedule = make_inference_schedule(k_max, nfe)
        assert len(schedule) == nfe
        for step in schedule:
            assert "tau" in step
            assert "signal_value" in step
            assert "step_value" in step


def test_shortcut_jepa_constructor():
    encoder = MockEncoder()
    predictor = ARPredictor(
        num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
        depth=2, heads=4, mlp_dim=512, dim_head=64,
    )
    action_encoder = Embedder(input_dim=10, emb_dim=192)
    flow_head = ShortcutFlowHead(dim=192, hidden_dim=512, k_max=8)
    model = ShortcutJEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        flow_head=flow_head,
        projector=torch.nn.Identity(),
        pred_proj=torch.nn.Identity(),
        k_max=8,
        target_space="latent",
    )
    assert model.k_max == 8
    assert model.target_space == "latent"
    assert model.sampling_nfe == 1


def test_nfe_1_calls_flow_head_once():
    flow_head = ShortcutFlowHead(dim=192, hidden_dim=512, k_max=8)
    call_count = {"count": 0}

    class TrackingFlowHead(torch.nn.Module):
        def forward(self, noisy, context, signal_idx, step_idx):
            call_count["count"] += 1
            return noisy

    B, T, D = 2, 3, 192
    model = ShortcutJEPA(
        encoder=MockEncoder(),
        predictor=ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=2, heads=4, mlp_dim=512, dim_head=64),
        action_encoder=Embedder(input_dim=10, emb_dim=192),
        flow_head=TrackingFlowHead(),
        k_max=8,
        target_space="latent",
    )
    context = torch.randn(B, T, D)
    noise = torch.randn_like(context)
    call_count["count"] = 0
    result = model.sample_next_from_context(context, noise=noise, nfe=1)
    assert call_count["count"] == 1, f"Expected 1 call, got {call_count['count']}"
    assert result.shape == (B, T, D)


def test_nfe_4_calls_flow_head_four_times():
    call_count = {"count": 0}

    class TrackingFlowHead(torch.nn.Module):
        def forward(self, noisy, context, signal_idx, step_idx):
            call_count["count"] += 1
            return noisy

    B, T, D = 2, 3, 192
    model = ShortcutJEPA(
        encoder=MockEncoder(),
        predictor=ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=2, heads=4, mlp_dim=512, dim_head=64),
        action_encoder=Embedder(input_dim=10, emb_dim=192),
        flow_head=TrackingFlowHead(),
        k_max=8,
        target_space="latent",
    )
    context = torch.randn(B, T, D)
    noise = torch.randn_like(context)
    call_count["count"] = 0
    result = model.sample_next_from_context(context, noise=noise, nfe=4)
    assert call_count["count"] == 4, f"Expected 4 calls, got {call_count['count']}"
    assert result.shape == (B, T, D)


def test_latent_and_delta_modes():
    B, T, D = 2, 3, 192
    for target_space in ["latent", "delta"]:
        model = ShortcutJEPA(
            encoder=MockEncoder(),
            predictor=ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=2, heads=4, mlp_dim=512, dim_head=64),
            action_encoder=Embedder(input_dim=10, emb_dim=192),
            flow_head=ShortcutFlowHead(dim=192, hidden_dim=512, k_max=8),
            k_max=8,
            target_space=target_space,
        )
        context = torch.randn(B, T, D)
        noise = torch.randn_like(context)
        if target_space == "delta":
            base_state = torch.randn(B, T, D)
            result = model.sample_next_from_context(context, base_state=base_state, noise=noise, nfe=1)
        else:
            result = model.sample_next_from_context(context, noise=noise, nfe=1)
        assert result.shape == (B, T, D)


def test_cost_shape():
    B, S = 2, 5
    model = ShortcutJEPA(
        encoder=MockEncoder(),
        predictor=ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=2, heads=4, mlp_dim=512, dim_head=64),
        action_encoder=Embedder(input_dim=10, emb_dim=192),
        flow_head=ShortcutFlowHead(dim=192, hidden_dim=512, k_max=8),
        k_max=8,
        target_space="latent",
    )
    model.planning_seed = 42
    model.configure_sampling(nfe=1, num_model_samples=1, seed=42)
    info_dict, action_candidates = make_info_dict(B, S)
    cost = model.get_cost(info_dict, action_candidates)
    assert cost.shape == (B, S), f"Expected ({B},{S}), got {cost.shape}"


def test_common_noise_deterministic():
    B, S = 2, 5
    model = ShortcutJEPA(
        encoder=MockEncoder(),
        predictor=ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=2, heads=4, mlp_dim=512, dim_head=64),
        action_encoder=Embedder(input_dim=10, emb_dim=192),
        flow_head=ShortcutFlowHead(dim=192, hidden_dim=512, k_max=8),
        k_max=8,
        target_space="latent",
    )
    model.planning_seed = 42
    model.configure_sampling(nfe=1, num_model_samples=1, seed=42)
    info_dict, action_candidates = make_info_dict(B, S)
    info_dict2 = {k: v.clone() for k, v in info_dict.items()}
    cost1 = model.get_cost(info_dict, action_candidates)
    cost2 = model.get_cost(info_dict2, action_candidates)
    assert torch.allclose(cost1, cost2), "get_cost should be deterministic for same inputs and seed"


def test_planning_seed_changes_cost():
    B, S = 2, 3
    model = ShortcutJEPA(
        encoder=MockEncoder(),
        predictor=ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=2, heads=4, mlp_dim=512, dim_head=64),
        action_encoder=Embedder(input_dim=10, emb_dim=192),
        flow_head=ShortcutFlowHead(dim=192, hidden_dim=512, k_max=8),
        k_max=8,
        target_space="latent",
    )
    info_dict1, action_candidates = make_info_dict(B, S)
    info_dict2 = {k: v.clone() for k, v in info_dict1.items()}
    model.planning_seed = 1
    model.configure_sampling(nfe=1, num_model_samples=1, seed=1)
    cost_a = model.get_cost(info_dict1, action_candidates)
    model.planning_seed = 2
    model.configure_sampling(nfe=1, num_model_samples=1, seed=2)
    cost_b = model.get_cost(info_dict2, action_candidates)
    assert not torch.allclose(cost_a, cost_b), "Different seeds should give different costs"


def test_rollout_key():
    B, S = 2, 3
    model = ShortcutJEPA(
        encoder=MockEncoder(),
        predictor=ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=2, heads=4, mlp_dim=512, dim_head=64),
        action_encoder=Embedder(input_dim=10, emb_dim=192),
        flow_head=ShortcutFlowHead(dim=192, hidden_dim=512, k_max=8),
        k_max=8,
        target_space="latent",
    )
    info = {
        "pixels": torch.randn(B, 3, 3, 224, 224),
    }
    action_sequence = torch.randn(B, S, 6, 10)
    result = model.rollout(info, action_sequence, history_size=3)
    assert "predicted_emb" in result, "rollout must contain predicted_emb key"


def test_predict_context_shape():
    model = ShortcutJEPA(
        encoder=MockEncoder(),
        predictor=ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=2, heads=4, mlp_dim=512, dim_head=64),
        action_encoder=Embedder(input_dim=10, emb_dim=192),
        flow_head=ShortcutFlowHead(dim=192, hidden_dim=512, k_max=8),
        k_max=8,
        target_space="latent",
    )
    B, T, D = 2, 3, 192
    emb = torch.randn(B, T, D)
    act_emb = torch.randn(B, T, D)
    context = model.predict_context(emb, act_emb)
    assert context.shape == (B, T, D)
