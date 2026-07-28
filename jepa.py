"""JEPA Implementation"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert "goal" in info_dict, "goal not in info_dict"
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)
        if "action" in goal:
            goal.pop("action")
        goal = self.encode(goal)

        goal_emb = goal["emb"]
        goal_emb = goal_emb.unsqueeze(1).expand(
            -1, action_candidates.shape[1], *([-1] * (goal_emb.ndim - 1))
        )
        info_dict["goal_emb"] = goal_emb
        info_dict = self.rollout(info_dict, action_candidates)
        cost = self.criterion(info_dict)
        return cost


class ShortcutJEPA(JEPA):
    def __init__(
        self, encoder, predictor, action_encoder, flow_head,
        projector=None, pred_proj=None, k_max=8, target_space="latent",
    ):
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
        self.flow_head = flow_head
        self.k_max = k_max
        self.target_space = target_space
        self.sampling_nfe = 1
        self.num_model_samples = 1
        self.planning_seed = 0

    def predict_context(self, emb, act_emb):
        context = self.predictor(emb, act_emb)
        context = self.pred_proj(rearrange(context, "b t d -> (b t) d"))
        return rearrange(context, "(b t) d -> b t d", b=emb.size(0))

    def flow_predict(self, noisy, context, signal_idx, step_idx):
        return self.flow_head(noisy, context, signal_idx, step_idx)

    def configure_sampling(self, nfe=None, num_model_samples=None, seed=None):
        if nfe is not None:
            self.sampling_nfe = nfe
        if num_model_samples is not None:
            assert num_model_samples == 1, (
                "Multi-sample planning is not implemented yet."
            )
            self.num_model_samples = num_model_samples
        if seed is not None:
            self.planning_seed = seed

    def sample_next_from_context(self, context, base_state=None, noise=None, nfe=None):
        nfe = nfe or self.sampling_nfe
        assert nfe <= self.k_max and (nfe & (nfe - 1)) == 0
        x = torch.randn_like(context) if noise is None else noise
        d = 1.0 / nfe
        step_value = int(math.log2(nfe))
        for i in range(nfe):
            tau = i / nfe
            signal_value = i * (self.k_max // nfe)
            signal_idx = torch.full(
                context.shape[:2], signal_value, device=context.device, dtype=torch.long
            )
            step_idx = torch.full(
                context.shape[:2], step_value, device=context.device, dtype=torch.long
            )
            pred_target = self.flow_predict(x, context, signal_idx, step_idx)
            velocity = (pred_target - x) / max(1e-4, 1.0 - tau)
            x = x + d * velocity
        if self.target_space == "latent":
            return x
        if self.target_space == "delta":
            assert base_state is not None
            return base_state + x
        raise ValueError(self.target_space)

    def rollout(self, info, action_sequence, history_size: int = 3, rollout_noise=None):
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            context = self.predict_context(emb_trunc, act_trunc)[:, -1:]
            noise_slice = (
                rollout_noise[:, t:t+1] if rollout_noise is not None else None
            )
            pred_emb = self.sample_next_from_context(
                context=context,
                base_state=emb[:, -1:],
                noise=noise_slice,
            )
            emb = torch.cat([emb, pred_emb], dim=1)
            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        context = self.predict_context(emb_trunc, act_trunc)[:, -1:]
        final_noise = (
            rollout_noise[:, n_steps:n_steps + 1]
            if rollout_noise is not None
            else None
        )
        pred_emb = self.sample_next_from_context(
            context=context,
            base_state=emb[:, -1:],
            noise=final_noise,
        )
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info

    def get_cost(self, info_dict, action_candidates):
        assert "goal" in info_dict, "goal not in info_dict"
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        B, S, T = action_candidates.shape[:3]

        goal = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                if k == "goal" and v.dim() < 5:
                    goal[k] = v
                else:
                    goal[k] = v[:, 0]
        goal["pixels"] = goal["goal"]
        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_"):]] = goal.pop(k)
        if "action" in goal:
            goal.pop("action")
        goal = self.encode(goal)
        info_dict["goal_emb"] = goal["emb"].unsqueeze(1).expand(B, S, -1, -1)

        latent_dim = goal["emb"].size(-1)
        n_steps = T - info_dict["pixels"].size(2)

        generator = torch.Generator(device=device)
        generator.manual_seed(self.planning_seed)
        noise = torch.randn(
            B, 1, n_steps + 1, latent_dim,
            generator=generator, device=device, dtype=goal["emb"].dtype,
        )
        noise = noise.expand(B, S, n_steps + 1, latent_dim)
        noise = rearrange(noise, "b s t d -> (b s) t d")

        info_dict = self.rollout(info_dict, action_candidates, rollout_noise=noise)
        cost = self.criterion(info_dict)
        return cost
