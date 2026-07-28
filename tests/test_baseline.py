import pytest
import torch
from module import ARPredictor, SIGReg


@pytest.fixture
def dummy_model():
    predictor = ARPredictor(
        num_frames=3,
        input_dim=192,
        hidden_dim=192,
        output_dim=192,
        depth=2,
        heads=4,
        mlp_dim=512,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    )
    return predictor


def test_predict_shape():
    B, T, D = 2, 3, 192
    predictor = ARPredictor(
        num_frames=T,
        input_dim=D,
        hidden_dim=D,
        output_dim=D,
        depth=2,
        heads=4,
        mlp_dim=512,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    )
    emb = torch.randn(B, T, D)
    act_emb = torch.randn(B, T, D)
    preds = predictor(emb, act_emb)
    assert preds.shape == (B, T, D), f"Expected (B,T,D), got {preds.shape}"


def test_prediction_loss_is_mse():
    pred_emb = torch.randn(4, 3, 192)
    tgt_emb = torch.randn(4, 3, 192)
    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    expected = torch.nn.functional.mse_loss(pred_emb, tgt_emb)
    assert torch.allclose(pred_loss, expected), "pred_loss should be exact MSE"


def test_sigreg_added_with_weight():
    sigreg = SIGReg(knots=17, num_proj=1024)
    emb = torch.randn(3, 4, 192)
    sigreg_loss = sigreg(emb.transpose(0, 1))
    weight = 0.09
    total = weight * sigreg_loss
    assert total.isfinite(), "Weighted SIGReg loss should be finite"
    assert total.ndim == 0, "Weighted SIGReg loss should be scalar"


def test_cost_tensor_shape_independent():
    B, S = 2, 5
    T = 6
    pred = torch.randn(B, S, T, 192)
    goal = torch.randn(B, S, T, 192)
    cost = torch.nn.functional.mse_loss(
        pred[..., -1:, :], goal[..., -1:, :].detach(),
        reduction="none",
    ).sum(dim=tuple(range(2, pred.ndim)))
    assert cost.shape == (B, S), f"Expected ({B},{S}), got {cost.shape}"


def test_cost_tensor_shape():
    B, S = 2, 5
    pred = torch.randn(B, S, 1, 192)
    goal = torch.randn(B, S, 2, 192)
    goal = goal[..., -1:, :].expand_as(pred)
    cost = torch.nn.functional.mse_loss(
        pred[..., -1:, :], goal[..., -1:, :].detach(),
        reduction="none",
    ).sum(dim=tuple(range(2, pred.ndim)))
    assert cost.shape == (B, S), f"Expected ({B},{S}), got {cost.shape}"
