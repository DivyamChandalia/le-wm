import json
import subprocess
import sys
import time
from pathlib import Path

import stable_worldmodel as swm
import torch


def limit_dataset(dataset, count, seed):
    count = min(count, len(dataset))
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:count].tolist()
    return torch.utils.data.Subset(dataset, indices)


def run_training(config_name, overrides, output_name):
    cmd = [
        sys.executable, "train.py",
        f"--config-name={config_name}",
        *overrides,
        f"output_model_name={output_name}",
    ]
    print(f"Running: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s")
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
    return result, elapsed


def benchmark_inference(model, device="cuda", history=3, latent_dim=192, horizon=5, warmup=20, timed_iters=100):
    model = model.to(device).eval()
    model.requires_grad_(False)
    B = 1

    for _ in range(warmup):
        emb = torch.randn(B, history, latent_dim, device=device)
        act_emb = torch.randn(B, history, latent_dim, device=device)
        if hasattr(model, "sample_next_from_context"):
            context = model.predict_context(emb, act_emb)[:, -1:]
            noise = torch.randn_like(context)
            model.sample_next_from_context(context=context, base_state=emb[:, -1:], noise=noise)
        else:
            model.predict(emb, act_emb)

    torch.cuda.synchronize()
    transitions = []
    five_step_times = []

    for _ in range(timed_iters):
        emb = torch.randn(B, history, latent_dim, device=device)
        act_emb = torch.randn(B, history, latent_dim, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for step in range(horizon):
            emb_trunc = emb[:, -history:]
            act_trunc = act_emb[:, -history:]
            if hasattr(model, "sample_next_from_context"):
                context = model.predict_context(emb_trunc, act_trunc)[:, -1:]
                noise = torch.randn_like(context)
                pred = model.sample_next_from_context(context=context, base_state=emb[:, -1:], noise=noise)
            else:
                pred = model.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
            act_emb = torch.cat([act_emb, torch.randn_like(act_emb[:, -1:])], dim=1)
            torch.cuda.synchronize()
            if step == 0:
                transitions.append(time.perf_counter() - t0)
        torch.cuda.synchronize()
        five_step_times.append(time.perf_counter() - t0)

    transition_ms = 1000 * torch.tensor(transitions[5:]).mean().item()
    five_step_ms = 1000 * torch.tensor(five_step_times[5:]).mean().item()
    trans_per_sec = 1000.0 / transition_ms
    return {
        "transition_ms": transition_ms,
        "five_step_ms": five_step_ms,
        "transitions_per_second": trans_per_sec,
    }


def benchmark():
    print("=" * 60)
    print("RTX 3050 Ti Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU benchmark.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    idle = torch.cuda.memory_allocated(0) / 1024**3
    print(f"Idle VRAM: {idle:.3f} GB")

    common = "data=pusht seed=3072 trainer.max_steps=100 trainer.accumulate_grad_batches=8".split()

    experiments = [
        {
            "name": "rtx3050ti_lewm",
            "config_name": "rtx3050ti",
            "overrides": ["model=lewm", "objective=mse"],
        },
        {
            "name": "rtx3050ti_flow",
            "config_name": "rtx3050ti",
            "overrides": ["model=shortcut_lewm", "objective=flow", "objective.k_max=4", "objective.target_space=latent"],
        },
        {
            "name": "rtx3050ti_shortcut",
            "config_name": "rtx3050ti",
            "overrides": ["model=shortcut_lewm", "objective=shortcut", "objective.k_max=4", "objective.target_space=latent", "objective.bootstrap_start_steps=10", "objective.alternate_batches=true"],
        },
    ]

    results = {}
    for exp in experiments:
        print(f"\n--- Training {exp['name']} ---")
        result, elapsed = run_training(exp["config_name"], common + exp["overrides"], exp["name"])
        training_info = {"training_time_s": elapsed, "returncode": result.returncode}

        try:
            model = swm.wm.utils.load_pretrained(exp["name"])
            model = model.to("cuda").eval()
            model.requires_grad_(False)
            if hasattr(model, "interpolate_pos_encoding"):
                model.interpolate_pos_encoding = True

            vram_before = torch.cuda.memory_allocated(0)
            inference = benchmark_inference(model, timed_iters=50)
            vram_after = torch.cuda.memory_allocated(0)
            inference["peak_vram_gb"] = vram_after / 1024**3
            training_info["inference"] = inference
        except Exception as e:
            print(f"Inference benchmark for {exp['name']} failed: {e}")
            training_info["inference"] = None

        peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"Peak VRAM during {exp['name']}: {peak_vram:.3f} GB")
        training_info["peak_vram_gb"] = peak_vram

        results[exp["name"]] = training_info

    results_file = Path("rtx3050ti_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    benchmark()
