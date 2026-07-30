# LeWM Flow-Matching + Shortcut-Forcing — Results

## Setup
- **Models**: ViT-tiny, 192-D latent, img_size=112, history_size=3, num_preds=1
- **Training**: 100 epochs, effective batch 128 (batch=32, accum=4), AdamW lr=5e-5, SIGReg weight=0.09, 16-mixed precision
- **Dataset**: PushT expert (500 episodes, ~50,500 frames, frameskip=5)
- **GPU**: RTX 3050 Ti Laptop 4 GB VRAM (peak ~1.08 GB allocated)
- **Checkpoints**: `~/.stable_worldmodel/checkpoints/<run_name>/`

## Offline Latent Dynamics Evaluation

64 held-out sequences, fixed noise seed 12345. Reported: latent MSE (↓), cosine similarity (↑).

| Model | Target Space | 1-step MSE | 3-step MSE | 5-step MSE | 1-step Cos | 5-step Cos |
|-------|-------------|-----------|-----------|-----------|-----------|-----------|
| LeWM (MSE baseline) | — | **0.008 ± 0.0004** | **0.036 ± 0.011** | **0.070 ± 0.015** | **0.996** | **0.955** |
| Flow-LeWM | latent | 0.317 ± 0.007 | 0.643 ± 0.003 | 0.816 ± 0.001 | 0.865 | 0.385 |
| Flow-LeWM | delta | 0.086 ± 0.012 | 0.230 ± 0.039 | 0.369 ± 0.046 | 0.942 | 0.765 |
| Shortcut-LeWM | delta | **0.084 ± 0.027** | **0.202 ± 0.063** | **0.332 ± 0.087** | **0.962** | **0.867** |

## Conditioning Dependence (1-step Shuffle Ratios)

Shuffle ratio >1 indicates the model uses that input. Higher = stronger conditioning.

| Model | Target Space | Action Shuffle Ratio | Context Shuffle Ratio |
|-------|-------------|---------------------|---------------------|
| LeWM (MSE baseline) | — | **6.3 ×** ✅ | **225.7 ×** ✅ |
| Flow-LeWM | latent | 1.07 × ❌ | 3.4 × ⚠️ |
| Flow-LeWM | delta | 1.00 × ❌ | 19.0 × ✅ |
| Shortcut-LeWM | delta | 1.00 × ❌ | 14.9 × ✅ |

## Key Findings

1. **Baseline MSE works well** — excellent latent dynamics (0.008 MSE, 0.996 cos) and strong conditioning on both actions (6.3×) and visual context (225×).

2. **Flow with `target_space=latent` fails** — predicts the SIGReg Gaussian marginal instead of conditioned dynamics. Action shuffle ratio ~1.07× (effectively ignores actions), context shuffle only 3.4×.

3. **`target_space=delta` helps but doesn't fix action conditioning** — switching to delta-space targets improves MSE 3–4× (0.086 vs 0.317) and context conditioning jumps to 19×. However, **action shuffle ratio remains ~1.0×** — the flow head ignores the ARPredictor context that carries action information.

4. **Shortcut-LeWM matches Flow-LeWM** — similar performance to flow delta, no improvement from shortcut bootstrapping. Same action conditioning collapse.

## Suspected Root Cause

The `ShortcutFlowHead` uses `noisy_proj(x) + context_proj(x)` — a residual addition that the MLP can learn to ignore. The noise projection dominates because it directly processes the target signal, while context is a small additive bias. The MSE baseline proves the ARPredictor *does* carry action information (6.3× shuffle ratio), so the bottleneck is in the flow head's ability to use that context.

**Recommended fix**: Concatenate `[noisy_proj; context_proj]` instead of adding, or use adaLN-style modulation in the flow head.

## Runtime & Memory

| Model | Peak GPU Allocated | Peak GPU Reserved |
|-------|-------------------|-------------------|
| LeWM (MSE) | ~1.08 GB | ~1.13 GB |
| Flow-LeWM | ~1.08 GB | ~1.13 GB |
| Shortcut-LeWM | ~1.08 GB | ~1.13 GB |
