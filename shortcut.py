import math
import torch


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def step_idx_from_nfe(nfe):
    return int(math.log2(nfe))


def sample_finest_flow_batch(target, k_max):
    noise = torch.randn_like(target)
    signal_idx = torch.randint(0, k_max, target.shape[:2], device=target.device)
    tau = signal_idx.float() / float(k_max)
    x_t = (1.0 - tau[..., None]) * noise + tau[..., None] * target
    step_idx = torch.full_like(signal_idx, int(math.log2(k_max)))
    weight = 0.9 * tau + 0.1
    return {
        "x_t": x_t,
        "noise": noise,
        "signal_idx": signal_idx,
        "step_idx": step_idx,
        "tau": tau,
        "weight": weight,
    }


def flow_xpred_loss(pred, true, weight):
    per_item = (pred - true).pow(2).mean(dim=-1)
    loss = (per_item * weight).mean()
    return loss


def sample_shortcut_grid(batch_shape, k_max, device):
    B, T = batch_shape
    K = k_max
    max_exp = int(math.log2(K))
    nfe_choices = [2**e for e in range(max_exp)]
    nfe_choices_tensor = torch.tensor(nfe_choices, device=device)
    nfe_indices = torch.randint(0, len(nfe_choices), (B, T), device=device)
    nfe = nfe_choices_tensor[nfe_indices]
    K_float = float(K)
    d = 1.0 / nfe.float()
    step_val = torch.log2(nfe.float()).long()
    rand_frac = torch.rand(B, T, device=device)
    j = (rand_frac * nfe.float()).long().clamp(0, nfe.max().item() - 1)
    signal_idx = (j.float() * (K_float / nfe.float())).long()
    signal_idx = signal_idx.clamp(0, K - 1)
    tau = signal_idx.float() / K_float
    tau_plus_d = tau + d
    valid = tau_plus_d <= 1.0
    return {
        "nfe": nfe,
        "d": d,
        "step_idx": step_val,
        "signal_idx": signal_idx,
        "tau": tau,
        "tau_plus_d": tau_plus_d,
        "valid": valid,
    }


def shortcut_bootstrap_loss(flow_head, x_t, context, signal_idx, step_idx, tau, d, k_max):
    pred_coarse = flow_head(x_t, context, signal_idx, step_idx)
    v_coarse = (pred_coarse - x_t) / (1.0 - tau).clamp_min(1e-4)[..., None]

    half_step_idx = step_idx + 1
    half_d = d / 2.0
    pred_half_1 = flow_head(x_t, context, signal_idx, half_step_idx)
    v1 = (pred_half_1 - x_t) / (1.0 - tau).clamp_min(1e-4)[..., None]
    x_mid = x_t + half_d[..., None] * v1

    signal_idx_mid = signal_idx + (half_d * float(k_max)).long()
    tau_mid = tau + half_d
    pred_half_2 = flow_head(x_mid, context, signal_idx_mid, half_step_idx)
    v2 = (pred_half_2 - x_mid) / (1.0 - tau_mid).clamp_min(1e-4)[..., None]

    v_target = ((v1 + v2) / 2.0).detach()
    per_item = (1.0 - tau).pow(2) * (v_coarse - v_target).pow(2).mean(dim=-1)
    weight = 0.9 * tau + 0.1
    loss = (per_item * weight).mean()
    return loss


def make_inference_schedule(k_max, nfe):
    assert is_power_of_two(nfe) and nfe <= k_max
    d = 1.0 / nfe
    schedule = []
    for i in range(nfe):
        tau = i / nfe
        signal_value = i * (k_max // nfe)
        step_value = int(math.log2(nfe))
        schedule.append({"tau": tau, "signal_value": signal_value, "step_value": step_value})
    return schedule
