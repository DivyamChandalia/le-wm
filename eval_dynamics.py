import json
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm


def limit_dataset(dataset, count, seed):
    count = min(count, len(dataset))
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:count].tolist()
    return torch.utils.data.Subset(dataset, indices)


def evaluate_latent_dynamics(model, dataset, horizons, noise_seed=12345, num_samples=64, device="cuda"):
    model = model.to(device).eval()
    model.requires_grad_(False)

    indices = limit_dataset(dataset, num_samples, 0)
    all_metrics = {}
    for h in horizons:
        all_metrics[h] = {"mse": [], "cos_sim": []}

    generator = torch.Generator(device=device)
    generator.manual_seed(noise_seed)

    for idx in range(len(indices)):
        row = indices[idx]
        pixels = torch.from_numpy(row["pixels"]).float().to(device)
        action = torch.from_numpy(row["action"]).float().to(device)
        T = pixels.size(0)
        info = {"pixels": pixels.unsqueeze(0), "action": action.unsqueeze(0)}
        info = model.encode(info)
        emb = info["emb"]
        act_emb = info["act_emb"]

        for h in horizons:
            if T <= h + 1:
                continue
            ctx_emb = emb[:, :3]
            ctx_act = act_emb[:, :3]
            pred_embs = []
            for t in range(h):
                noise = torch.randn(1, 1, emb.size(-1), generator=generator, device=device)
                if hasattr(model, "sample_next_from_context"):
                    context = model.predict_context(ctx_emb, ctx_act)[:, -1:]
                    pred = model.sample_next_from_context(
                        context=context,
                        base_state=ctx_emb[:, -1:],
                        noise=noise,
                    )
                else:
                    pred = model.predict(ctx_emb, ctx_act)[:, -1:]
                pred_embs.append(pred)
                ctx_emb = torch.cat([ctx_emb, pred], dim=1)
                next_act = act_emb[:, 3 + t:4 + t] if 3 + t < act_emb.size(1) else act_emb[:, -1:]
                ctx_act = torch.cat([ctx_act, next_act], dim=1)

            pred_embs = torch.cat(pred_embs, dim=1)
            true_embs = emb[:, 3:3 + h]
            if true_embs.size(1) == 0:
                continue
            mse = (pred_embs - true_embs).pow(2).mean().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                pred_embs.reshape(-1, pred_embs.size(-1)),
                true_embs.reshape(-1, true_embs.size(-1)),
            ).mean().item()
            all_metrics[h]["mse"].append(mse)
            all_metrics[h]["cos_sim"].append(cos_sim)

    results = {}
    for h in horizons:
        if all_metrics[h]["mse"]:
            results[h] = {
                "mse_mean": np.mean(all_metrics[h]["mse"]),
                "mse_std": np.std(all_metrics[h]["mse"]),
                "cos_sim_mean": np.mean(all_metrics[h]["cos_sim"]),
                "cos_sim_std": np.std(all_metrics[h]["cos_sim"]),
            }
    return results


def evaluate_conditioning_shuffle(model, dataset, horizons=[1, 3, 5], noise_seed=12345, num_samples=32, device="cuda"):
    model = model.to(device).eval()
    model.requires_grad_(False)

    indices = limit_dataset(dataset, num_samples, 0)
    normal_losses = {h: [] for h in horizons}
    shuffle_action_losses = {h: [] for h in horizons}
    shuffle_context_losses = {h: [] for h in horizons}

    generator = torch.Generator(device=device)
    generator.manual_seed(noise_seed)

    for idx in range(len(indices)):
        row = indices[idx]
        pixels = torch.from_numpy(row["pixels"]).float().to(device)
        action = torch.from_numpy(row["action"]).float().to(device)
        T = pixels.size(0)
        info = {"pixels": pixels.unsqueeze(0), "action": action.unsqueeze(0)}
        info = model.encode(info)
        emb = info["emb"]
        act_emb = info["act_emb"]

        for h in horizons:
            if T <= h + 1:
                continue
            ctx_emb = emb[:, :3]
            ctx_act = act_emb[:, :3]
            true_embs = emb[:, 3:3 + h]
            if true_embs.size(1) == 0:
                continue

            if hasattr(model, "sample_next_from_context"):
                context = model.predict_context(ctx_emb, ctx_act)[:, -1:]
                noise = torch.randn(1, 1, emb.size(-1), generator=generator, device=device)
                pred = model.sample_next_from_context(context=context, base_state=ctx_emb[:, -1:], noise=noise)
            else:
                pred = model.predict(ctx_emb, ctx_act)[:, -1:]
            normal_loss = (pred - true_embs[:, :1]).pow(2).mean().item()
            normal_losses[h].append(normal_loss)

            act_shuffled = act_emb[:, torch.randperm(act_emb.size(1))]
            ctx_act_shuf = act_shuffled[:, :3]
            if hasattr(model, "sample_next_from_context"):
                context_shuf = model.predict_context(ctx_emb, ctx_act_shuf)[:, -1:]
                pred = model.sample_next_from_context(context=context_shuf, base_state=ctx_emb[:, -1:], noise=noise)
            else:
                pred = model.predict(ctx_emb, ctx_act_shuf)[:, -1:]
            shuffle_action_loss = (pred - true_embs[:, :1]).pow(2).mean().item()
            shuffle_action_losses[h].append(shuffle_action_loss)

            emb_shuffled = emb[:, torch.randperm(emb.size(1))]
            ctx_emb_shuf = emb_shuffled[:, :3]
            if hasattr(model, "sample_next_from_context"):
                context_shuf = model.predict_context(ctx_emb_shuf, ctx_act)[:, -1:]
                pred = model.sample_next_from_context(context=context_shuf, base_state=emb_shuffled[:, -1:], noise=noise)
            else:
                pred = model.predict(ctx_emb_shuf, ctx_act)[:, -1:]
            shuffle_context_loss = (pred - true_embs[:, :1]).pow(2).mean().item()
            shuffle_context_losses[h].append(shuffle_context_loss)

    results = {}
    for h in horizons:
        if normal_losses[h]:
            normal = np.mean(normal_losses[h])
            shuffle_act = np.mean(shuffle_action_losses[h])
            shuffle_ctx = np.mean(shuffle_context_losses[h])
            results[h] = {
                "normal_loss": normal,
                "shuffled_action_loss": shuffle_act,
                "shuffled_context_loss": shuffle_ctx,
                "action_shuffle_ratio": shuffle_act / normal if normal > 0 else 1.0,
                "context_shuffle_ratio": shuffle_ctx / normal if normal > 0 else 1.0,
            }
    return results


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = swm.policy.AutoCostModel(cfg.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)

    if hasattr(model, "configure_sampling"):
        sampling_cfg = cfg.get("sampling", {})
        model.configure_sampling(
            nfe=sampling_cfg.get("nfe", 1),
            num_model_samples=sampling_cfg.get("num_model_samples", 1),
            seed=sampling_cfg.get("seed", 0),
        )

    horizons = [1, 3, 5]
    print("Evaluating latent dynamics...")
    dynamics_results = evaluate_latent_dynamics(
        model, dataset, horizons=horizons, noise_seed=12345, num_samples=64, device=device,
    )
    print("Dynamics results:", json.dumps(dynamics_results, indent=2, default=str))

    print("Evaluating conditioning shuffle...")
    shuffle_results = evaluate_conditioning_shuffle(
        model, dataset, horizons=horizons, noise_seed=12345, num_samples=32, device=device,
    )
    print("Conditioning results:", json.dumps(shuffle_results, indent=2, default=str))

    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent if cfg.policy != "random" else Path.cwd()
    results_file = results_path / "dynamics_eval.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump({"dynamics": dynamics_results, "conditioning": shuffle_results}, f, indent=2, default=str)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    run()
