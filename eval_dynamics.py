import json
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from utils import get_column_normalizer, get_img_preprocessor


def collate_sequences(dataset, indices, history_size, device):
    B = len(indices)
    n_steps = history_size + 5 + 1
    pixels_list, action_list = [], []
    for idx in indices:
        row = dataset[idx]
        pixels_list.append(torch.from_numpy(row["pixels"][:n_steps]).float())
        action_list.append(torch.from_numpy(row["action"][:n_steps]).float())
    pixels = torch.stack(pixels_list).to(device)
    action = torch.stack(action_list).to(device)
    return pixels, action


def evaluate_latent_dynamics(model, dataset, horizons, noise_seed=12345, num_samples=64, device="cuda", history_size=3):
    model = model.to(device).eval()
    model.requires_grad_(False)

    all_metrics = {}
    for h in horizons:
        all_metrics[h] = {"mse": [], "cos_sim": []}

    generator = torch.Generator(device=device)
    generator.manual_seed(noise_seed)

    rnd = np.random.RandomState(0)
    total = len(dataset)
    batch_size = min(32, num_samples)
    for start in range(0, num_samples, batch_size):
        batch_indices = rnd.randint(0, total, batch_size).tolist()
        pixels, action = collate_sequences(dataset, batch_indices, history_size, device)
        B = pixels.size(0)
        info = {"pixels": pixels, "action": action}
        info = model.encode(info)
        emb = info["emb"]
        act_emb = info["act_emb"]

        for h in horizons:
            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]
            pred_embs = []
            for t in range(h):
                noise = torch.randn(B, 1, emb.size(-1), generator=generator, device=device)
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
                next_act = act_emb[:, history_size + t:history_size + t + 1]
                ctx_act = torch.cat([ctx_act, next_act], dim=1)

            pred_embs = torch.cat(pred_embs, dim=1)
            true_embs = emb[:, history_size:history_size + h]
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


def evaluate_conditioning_shuffle(model, dataset, horizons=[1], noise_seed=12345, num_samples=32, device="cuda", history_size=3):
    model = model.to(device).eval()
    model.requires_grad_(False)

    normal_losses = {h: [] for h in horizons}
    shuffle_action_losses = {h: [] for h in horizons}
    shuffle_context_losses = {h: [] for h in horizons}

    generator = torch.Generator(device=device)
    generator.manual_seed(noise_seed)

    rnd = np.random.RandomState(0)
    total = len(dataset)
    batch_size = min(32, num_samples)
    for start in range(0, num_samples, batch_size):
        batch_indices = rnd.randint(0, total, batch_size).tolist()
        pixels, action = collate_sequences(dataset, batch_indices, history_size, device)
        B = pixels.size(0)
        info = {"pixels": pixels, "action": action}
        info = model.encode(info)
        emb = info["emb"]
        act_emb = info["act_emb"]

        for h in horizons:
            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]
            true_embs = emb[:, history_size:history_size + h]

            noise = torch.randn(B, 1, emb.size(-1), generator=generator, device=device)

            if hasattr(model, "sample_next_from_context"):
                context = model.predict_context(ctx_emb, ctx_act)[:, -1:]
                pred = model.sample_next_from_context(
                    context=context, base_state=ctx_emb[:, -1:], noise=noise
                )
            else:
                pred = model.predict(ctx_emb, ctx_act)[:, -1:]
            normal_loss = (pred - true_embs[:, :1]).pow(2).mean().item()
            normal_losses[h].append(normal_loss)

            perm = torch.randperm(B, device=device)
            ctx_act_shuf = ctx_act[perm]
            if hasattr(model, "sample_next_from_context"):
                context_shuf = model.predict_context(ctx_emb, ctx_act_shuf)[:, -1:]
                pred = model.sample_next_from_context(
                    context=context_shuf, base_state=ctx_emb[:, -1:], noise=noise
                )
            else:
                pred = model.predict(ctx_emb, ctx_act_shuf)[:, -1:]
            shuffle_action_loss = (pred - true_embs[:, :1]).pow(2).mean().item()
            shuffle_action_losses[h].append(shuffle_action_loss)

            perm = torch.randperm(B, device=device)
            ctx_emb_shuf = ctx_emb[perm]
            if hasattr(model, "sample_next_from_context"):
                context_shuf = model.predict_context(ctx_emb_shuf, ctx_act)[:, -1:]
                pred = model.sample_next_from_context(
                    context=context_shuf, base_state=ctx_emb_shuf[:, -1:], noise=noise
                )
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

    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.eval.img_size)]
    for col in cfg.dataset.keys_to_cache:
        if col.startswith("pixels"):
            continue
        normalizer = get_column_normalizer(dataset, col, col)
        transforms.append(normalizer)
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

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

    horizons = [1]
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
