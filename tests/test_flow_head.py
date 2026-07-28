import pytest
import torch

from module import ShortcutFlowHead


@pytest.fixture
def head():
    return ShortcutFlowHead(dim=192, hidden_dim=512, k_max=8)


def test_output_shape(head):
    B, T, D = 2, 3, 192
    noisy_target = torch.randn(B, T, D)
    context = torch.randn(B, T, D)
    signal_idx = torch.randint(0, 8, (B, T))
    step_idx = torch.randint(0, 3, (B, T))
    out = head(noisy_target, context, signal_idx, step_idx)
    assert out.shape == (B, T, D), f"Expected (B,T,D), got {out.shape}"


def test_forward_backward(head):
    B, T, D = 2, 3, 192
    noisy_target = torch.randn(B, T, D, requires_grad=True)
    context = torch.randn(B, T, D)
    signal_idx = torch.randint(0, 8, (B, T))
    step_idx = torch.randint(0, 3, (B, T))
    out = head(noisy_target, context, signal_idx, step_idx)
    loss = out.sum()
    loss.backward()
    assert noisy_target.grad is not None, "Gradient should flow to noisy_target"
    assert noisy_target.grad.isfinite().all(), "All gradients should be finite"


def test_k_max_assertion():
    with pytest.raises(AssertionError):
        ShortcutFlowHead(dim=192, hidden_dim=512, k_max=7)
    with pytest.raises(AssertionError):
        ShortcutFlowHead(dim=192, hidden_dim=512, k_max=0)
    with pytest.raises(AssertionError):
        ShortcutFlowHead(dim=192, hidden_dim=512, k_max=-1)
