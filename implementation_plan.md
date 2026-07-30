Flow-Matching + Shortcut-Forcing LeWorldModel

Implementation, Testing, and Evaluation Canvas

**Primary engineering target**

Extend the existing LeWM repository with Flow-LeWM and Shortcut-LeWM while keeping the original LeWM training, rollout, checkpoint, and planning paths unchanged.

| **Field**              | **Value**                                               |
| ---------------------- | ------------------------------------------------------- |
| Repository             | lucas-maes/le-wm                                        |
| Baseline snapshot      | main at commit 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac |
| Primary task for pilot | PushT                                                   |
| Full benchmark tasks   | PushT, TwoRoom, Cube, Reacher                           |
| Local diagnostic GPU   | RTX 3050 Ti Laptop, 4 GB VRAM                           |
| Final model variants   | LeWM, Flow-LeWM, Shortcut-LeWM                          |

## Document use

- Give this document directly to a coding model or engineer.
- Complete stages in order; do not implement the full benchmark in one pass.
- Use the \[ \] items as an editable progress checklist.
- Record deviations, failed tests, and measured results in the blank Notes fields.

This is an editable DOCX. All text, tables, checklists, commands, and stage notes can be changed in Word, LibreOffice, or Google Docs.

# Contents

- Scope and non-negotiable rules
- Existing repository architecture
- Target model definitions
- Stage 0 - Baseline lock and regression tests
- Stage 1 - ShortcutFlowHead
- Stage 2 - Flow and shortcut mathematics
- Stage 3 - ShortcutJEPA model API
- Stage 4 - Flow rollout
- Stage 5 - Deterministic stochastic planning for CEM
- Stage 6 - Hydra model configuration
- Stage 7 - Objective configuration
- Stage 8 - Training config composition
- Stage 9 - Training objective dispatch
- Stage 10 - Evaluation sampling config
- Stage 11 - Evaluation integration
- Stage 12 - Required unit and integration tests
- Stage 13 - Smoke training
- Stage 13A - RTX 3050 Ti apples-to-apples comparison
- Stage 14 - Offline dynamics evaluation
- Stage 15 - Conditioning diagnostics
- Stage 16 - PushT pilot
- Stage 17 - NFE sweep
- Stage 18 - Failure-directed decision tree
- Stage 19 - Full benchmark
- Stage 20 - Final reporting
- File manifest and final acceptance criteria

# 1\. Scope and Non-Negotiable Rules

**Research question**

Can a compact, end-to-end JEPA world model replace deterministic next-latent MSE with a conditional stochastic latent flow, while shortcut forcing preserves the low inference cost needed for CEM planning?

## Models to support

| **Variant**   | **Training objective**                          | **Next-latent inference**                |
| ------------- | ----------------------------------------------- | ---------------------------------------- |
| LeWM          | Deterministic next-latent MSE + SIGReg          | One AR predictor call                    |
| Flow-LeWM     | Conditional flow x-prediction + SIGReg          | Full K-step latent flow                  |
| Shortcut-LeWM | Flow x-prediction + shortcut bootstrap + SIGReg | Selectable 1, 2, 4, or 8 flow-head calls |

## Must remain unchanged

\[ \] Do not modify stable-worldmodel.

\[ \] Do not replace CEM in the primary comparison.

\[ \] Do not change the visual encoder architecture for matched runs.

\[ \] Do not remove SIGReg.

\[ \] Do not add an EMA target encoder.

\[ \] Do not change the final latent-goal MSE criterion for the primary control comparison.

\[ \] Do not combine this work with temporal skipping or Fast-LeWM-style multi-environment-step prediction.

\[ \] Do not claim benchmark superiority from local or single-seed diagnostics.

## Implementation discipline

\[ \] Complete stages in order.

\[ \] Run tests at the end of every stage.

\[ \] Commit after each working stage.

\[ \] Keep the original JEPA class and original MSE path available.

\[ \] Make stochastic planning deterministic for fixed inputs and seed.

\[ \] Compute the expensive temporal context predictor once per simulated environment step, not once per flow substep.

# 2\. Existing Repository Architecture

| **File**                     | **Current responsibility**                                          | **Planned treatment**                               |
| ---------------------------- | ------------------------------------------------------------------- | --------------------------------------------------- |
| train.py                     | Dataset loading, model construction, LeWM loss, Lightning trainer   | Add objective dispatch; preserve exact MSE branch   |
| jepa.py                      | Encoding, prediction, rollout, latent goal cost                     | Keep JEPA unchanged; add ShortcutJEPA               |
| module.py                    | SIGReg, transformer blocks, action embedder, MLP, ARPredictor       | Keep ARPredictor unchanged; add ShortcutFlowHead    |
| eval.py                      | Checkpoint loading, CEM policy construction, environment evaluation | Only configure sampling for flow-capable models     |
| config/train/model/lewm.yaml | Original LeWM model graph                                           | Do not edit unless unavoidable                      |
| config/train/lewm.yaml       | Global train defaults                                               | Add objective config composition                    |
| config/eval/\*.yaml          | Task and planner settings                                           | Keep task settings unchanged for primary comparison |

**Baseline data alignment**

With history_size=3 and num_preds=1, the encoded sequence contains four states. Context positions z0,z1,z2 with actions a0,a1,a2 predict targets z1,z2,z3.

# 3\. Target Model Definitions

## Shared encoder and context network

z_t = E(o_t) # 192-D LeWM latent  
h_i = H(z_<=i, a_<=i) # existing ARPredictor  
next_latent = transition(h_i, noise) # deterministic or flow-based

- The ViT encoder, projector, action encoder, and ARPredictor architecture remain matched across models.
- Flow-LeWM and Shortcut-LeWM add a small flow head conditioned on the ARPredictor context.
- The flow head predicts the clean target latent (x-prediction), not raw velocity.

## Target spaces

| **Config**           | **Definition**                    | **Use**                                             |
| -------------------- | --------------------------------- | --------------------------------------------------- |
| target_space: latent | y = z_(t+1)                       | Primary experiment                                  |
| target_space: delta  | y = z_(t+1) - z_t; output z_t + y | Fallback if direct latent flow ignores conditioning |

**Known research risk**

SIGReg encourages the latent marginal to resemble an isotropic Gaussian. A Gaussian-noise-to-latent flow may learn the marginal while underusing state/action conditioning. Conditioning-shuffle tests and the delta-space fallback are mandatory.

# Stage 0 - Baseline Lock and Regression Tests

**Stage goal**

Prove the original LeWM path remains unchanged before adding flow code.

| **Files**              | **Action**              |
| ---------------------- | ----------------------- |
| tests/test_baseline.py | Create                  |
| train.py               | Read only in this stage |
| jepa.py                | Read only in this stage |

## Required tests

\[ \] JEPA.predict returns shape (B,T,D).

\[ \] The original prediction loss is exactly mean((pred_emb - tgt_emb)^2).

\[ \] SIGReg is added with the existing configured weight.

\[ \] For fixed model and inputs, JEPA.get_cost returns the same result twice.

\[ \] The final cost tensor has shape (B,S).

\# Existing baseline formula - do not change  
pred_loss = (pred_emb - tgt_emb).pow(2).mean()  
sigreg_loss = self.sigreg(emb.transpose(0, 1))  
loss = pred_loss + cfg.loss.sigreg.weight \* sigreg_loss

\[ \] Run: pytest tests/test_baseline.py

\[ \] Save expected numeric outputs or a fixed-seed regression fixture.

\[ \] Commit: test: lock baseline LeWM behavior

Notes / results: \_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**_

# Stage 1 - Add ShortcutFlowHead

**Stage goal**

Add the small conditional flow head without changing ARPredictor.

| **Files**               | **Action** |
| ----------------------- | ---------- |
| module.py               | Modify     |
| tests/test_flow_head.py | Create     |

## Exact code change

- Add import math.
- Append ShortcutFlowHead after ARPredictor.
- Do not edit ARPredictor.

class ShortcutFlowHead(nn.Module):  
def \__init_\_(self, dim: int, hidden_dim: int = 2048, k_max: int = 8):  
super().\__init_\_()  
assert k_max > 0 and (k_max & (k_max - 1)) == 0  
self.dim = dim  
self.k_max = k_max  
self.max_step_idx = int(math.log2(k_max))  
self.noisy_proj = nn.Linear(dim, dim)  
self.context_proj = nn.Linear(dim, dim)  
self.signal_embed = nn.Embedding(k_max + 1, dim)  
self.step_embed = nn.Embedding(self.max_step_idx + 1, dim)  
self.net = nn.Sequential(  
nn.LayerNorm(dim),  
nn.Linear(dim, hidden_dim),  
nn.GELU(),  
nn.Linear(hidden_dim, dim),  
)  
<br/>def forward(self, noisy_target, context, signal_idx, step_idx):  
x = self.noisy_proj(noisy_target)  
x = x + self.context_proj(context)  
x = x + self.signal_embed(signal_idx)  
x = x + self.step_embed(step_idx)  
return self.net(x)

\[ \] Test B=2,T=3,D=192 output shape.

\[ \] Test k_max=7 raises an assertion.

\[ \] Test forward/backward on CPU.

\[ \] Commit: feat: add conditional shortcut flow head

# Stage 2 - Add Flow and Shortcut Mathematics

**Stage goal**

Keep all schedule, target, bootstrap, and inference math in a separate module.

| **Files**              | **Action**     |
| ---------------------- | -------------- |
| shortcut.py            | Create         |
| tests/test_shortcut.py | Begin creating |

## Functions to implement

| **Function**                                   | **Purpose**                                                |
| ---------------------------------------------- | ---------------------------------------------------------- |
| is_power_of_two(n)                             | Validate K and NFE values                                  |
| step_idx_from_nfe(nfe)                         | Map 1,2,4,8 to 0,1,2,3                                     |
| sample_finest_flow_batch(target,k_max)         | Construct Gaussian-to-target interpolation at finest level |
| flow_xpred_loss(pred,true,weight)              | Weighted clean-target MSE                                  |
| sample_shortcut_grid(batch_shape,k_max,device) | Sample coarse shortcut step and reachable tau grid         |
| shortcut_bootstrap_loss(...)                   | Match one coarse step to two half steps                    |
| make_inference_schedule(k_max,nfe)             | Return discrete signal indices and dt                      |

## Finest flow batch

noise = torch.randn_like(target)  
signal_idx = torch.randint(0, k_max, target.shape\[:2\], device=target.device)  
tau = signal_idx.float() / float(k_max)  
x_t = (1.0 - tau\[..., None\]) \* noise + tau\[..., None\] \* target  
step_idx = torch.full_like(signal_idx, int(math.log2(k_max)))  
weight = 0.9 \* tau + 0.1

## Weighted x-prediction loss

per_item = (pred_target - true_target).pow(2).mean(dim=-1)  
loss = (per_item \* weight).mean()

## Shortcut grid

- For k_max=8, bootstrap NFE choices are 1, 2, or 4. The finest level 8 remains empirical flow supervision.
- For NFE=K, d=1/K and tau=j/K for integer j in \[0,K-1\].
- signal_idx = j \* (k_max // K).

## Bootstrap calculation

pred_coarse = flow_head(x_t, context, signal_idx, step_idx)  
v_coarse = (pred_coarse - x_t) / (1.0 - tau).clamp_min(1e-4)\[..., None\]  
<br/>half_step_idx = step_idx + 1  
half_d = d / 2.0  
pred_half_1 = flow_head(x_t, context, signal_idx, half_step_idx)  
v1 = (pred_half_1 - x_t) / (1.0 - tau).clamp_min(1e-4)\[..., None\]  
x_mid = x_t + half_d\[..., None\] \* v1  
<br/>signal_idx_mid = signal_idx + (half_d \* float(k_max)).long()  
tau_mid = tau + half_d  
pred_half_2 = flow_head(x_mid, context, signal_idx_mid, half_step_idx)  
v2 = (pred_half_2 - x_mid) / (1.0 - tau_mid).clamp_min(1e-4)\[..., None\]  
<br/>v_target = ((v1 + v2) / 2.0).detach()  
per_item = (1.0 - tau).pow(2) \* (v_coarse - v_target).pow(2).mean(dim=-1)  
loss = (per_item \* (0.9 \* tau + 0.1)).mean()

\[ \] All schedule tensors have shape (B,T).

\[ \] All losses are finite on random CPU tensors.

\[ \] No gradient flows through v_target.

\[ \] Validate tau+d <= 1 for all sampled items.

\[ \] Commit: feat: add flow and shortcut objectives

# Stage 3 - Add ShortcutJEPA Model API

**Stage goal**

Add a new model class that reuses JEPA encoding/criterion behavior and adds flow-specific methods.

| **Files**              | **Action**                       |
| ---------------------- | -------------------------------- |
| jepa.py                | Modify by appending ShortcutJEPA |
| tests/test_shortcut.py | Extend                           |

## Constructor

class ShortcutJEPA(JEPA):  
def \__init_\_(self, encoder, predictor, action_encoder, flow_head,  
projector=None, pred_proj=None, k_max=8, target_space="latent"):  
super().\__init_\_(encoder, predictor, action_encoder, projector, pred_proj)  
self.flow_head = flow_head  
self.k_max = k_max  
self.target_space = target_space  
self.sampling_nfe = 1  
self.num_model_samples = 1  
self.planning_seed = 0

## Methods to add

| **Method**                                               | **Exact behavior**                                                  |
| -------------------------------------------------------- | ------------------------------------------------------------------- |
| predict_context(emb,act_emb)                             | Run existing predictor + pred_proj; return (B,T,D) context          |
| flow_predict(noisy,context,signal_idx,step_idx)          | Delegate to flow_head                                               |
| configure_sampling(nfe,num_model_samples,seed)           | Validate and store inference settings                               |
| sample_next_from_context(context,base_state,noise,nfe)   | Run K flow-head Euler steps                                         |
| rollout(info,action_sequence,history_size,rollout_noise) | Autoregressive latent rollout                                       |
| get_cost(info,action_candidates)                         | Encode goal, generate fixed noise, rollout, use inherited criterion |

## Context method

def predict_context(self, emb, act_emb):  
context = self.predictor(emb, act_emb)  
context = self.pred_proj(rearrange(context, "b t d -> (b t) d"))  
return rearrange(context, "(b t) d -> b t d", b=emb.size(0))

## Sampling method

def sample_next_from_context(self, context, base_state=None, noise=None, nfe=None):  
nfe = nfe or self.sampling_nfe  
assert nfe <= self.k_max and (nfe & (nfe - 1)) == 0  
x = torch.randn_like(context) if noise is None else noise  
d = 1.0 / nfe  
step_value = int(math.log2(nfe))  
for i in range(nfe):  
tau = i / nfe  
signal_value = i \* (self.k_max // nfe)  
signal_idx = torch.full(context.shape\[:2\], signal_value, device=context.device, dtype=torch.long)  
step_idx = torch.full(context.shape\[:2\], step_value, device=context.device, dtype=torch.long)  
pred_target = self.flow_predict(x, context, signal_idx, step_idx)  
velocity = (pred_target - x) / max(1e-4, 1.0 - tau)  
x = x + d \* velocity  
if self.target_space == "latent":  
return x  
if self.target_space == "delta":  
assert base_state is not None  
return base_state + x  
raise ValueError(self.target_space)

\[ \] Do not remove or rename JEPA.predict.

\[ \] Test nfe=1 invokes flow_head once.

\[ \] Test nfe=4 invokes flow_head four times.

\[ \] Test latent and delta output modes.

\[ \] Commit: feat: add ShortcutJEPA sampling API

# Stage 4 - Add Flow Rollout

**Stage goal**

Make ShortcutJEPA satisfy the same rollout contract used by stable-worldmodel.

| **Files**              | **Action**               |
| ---------------------- | ------------------------ |
| jepa.py                | Add ShortcutJEPA.rollout |
| tests/test_shortcut.py | Add rollout tests        |

- Copy the outer structure of JEPA.rollout.
- Encode the initial history once.
- Flatten environment batch B and CEM candidates S.
- At each future environment step, compute ARPredictor context once.
- Use the small flow head NFE times to generate one next latent.
- Append the generated latent and next action.
- Retain the baseline final extra prediction before scoring.

for t in range(n_steps):  
act_emb = self.action_encoder(act)  
emb_trunc = emb\[:, -history_size:\]  
act_trunc = act_emb\[:, -history_size:\]  
context = self.predict_context(emb_trunc, act_trunc)\[:, -1:\]  
pred_emb = self.sample_next_from_context(  
context=context,  
base_state=emb\[:, -1:\],  
noise=rollout_noise\[:, t:t+1\] if rollout_noise is not None else None,  
)  
emb = torch.cat(\[emb, pred_emb\], dim=1)  
act = torch.cat(\[act, act_future\[:, t:t+1\]\], dim=1)

**Performance invariant**

ARPredictor must run once per generated environment transition. Only ShortcutFlowHead repeats across NFE substeps.

\[ \] Rollout output key remains info\["predicted_emb"\].

\[ \] Output dimensions match the existing criterion.

\[ \] Forward-hook test verifies one ARPredictor call per environment transition.

\[ \] Commit: feat: add shortcut latent rollout

# Stage 5 - Make Stochastic Planning Deterministic for CEM

**Stage goal**

Use common random numbers so candidate rankings and repeated CEM iterations see a stable cost function.

| **Files**              | **Action**                                                       |
| ---------------------- | ---------------------------------------------------------------- |
| jepa.py                | Override ShortcutJEPA.get_cost and extend rollout noise handling |
| tests/test_shortcut.py | Add deterministic-cost tests                                     |

- Create a new torch.Generator inside every get_cost call.
- Seed it with self.planning_seed.
- Sample noise by environment batch, model sample, and future transition.
- Expand identical transition noise across the CEM candidate dimension S.
- For the first implementation, support num_model_samples=1 only and assert it.

generator = torch.Generator(device=device)  
generator.manual_seed(self.planning_seed)  
noise = torch.randn(  
B, 1, rollout_steps, latent_dim,  
generator=generator, device=device, dtype=goal_emb.dtype,  
)  
\# Expand the same noise across candidate plans S.  
noise = noise.expand(B, S, rollout_steps, latent_dim)  
noise = rearrange(noise, "b s t d -> (b s) t d")

\[ \] Two get_cost calls with the same inputs and seed are torch.allclose.

\[ \] Changing planning_seed changes flow predictions.

\[ \] Candidate plans share transition noise at each timestep.

\[ \] Cost shape remains (B,S).

\[ \] Commit: fix: stabilize stochastic CEM costs with common noise

# Stage 6 - Add Hydra Model Configuration

**Stage goal**

Instantiate ShortcutJEPA while leaving the original LeWM model config untouched.

| **Files**                             | **Action**      |
| ------------------------------------- | --------------- |
| config/train/model/shortcut_lewm.yaml | Create          |
| config/train/model/lewm.yaml          | Leave unchanged |

\_target_: jepa.ShortcutJEPA  
<br/>k_max: \${objective.k_max}  
target_space: \${objective.target_space}  
<br/>encoder:  
\_target_: stable_pretraining.backbone.utils.vit_hf  
size: tiny  
patch_size: 14  
image_size: \${img_size}  
pretrained: false  
use_mask_token: false  
<br/>predictor:  
\_target_: module.ARPredictor  
num_frames: \${history_size}  
input_dim: \${embed_dim}  
hidden_dim: \${embed_dim}  
output_dim: \${embed_dim}  
depth: 6  
heads: 16  
mlp_dim: 2048  
dim_head: 64  
dropout: 0.1  
emb_dropout: 0.0  
<br/>flow_head:  
\_target_: module.ShortcutFlowHead  
dim: \${embed_dim}  
hidden_dim: 2048  
k_max: \${objective.k_max}  
<br/>action_encoder:  
\_target_: module.Embedder  
input_dim: ???  
emb_dim: \${embed_dim}  
<br/>projector:  
\_target_: module.MLP  
input_dim: \${embed_dim}  
output_dim: \${embed_dim}  
hidden_dim: 2048  
norm_fn:  
\_target_: torch.nn.BatchNorm1d  
\_partial_: true  
<br/>pred_proj:  
\_target_: module.MLP  
input_dim: \${embed_dim}  
output_dim: \${embed_dim}  
hidden_dim: 2048  
norm_fn:  
\_target_: torch.nn.BatchNorm1d  
\_partial_: true

\[ \] Hydra can instantiate model=shortcut_lewm.

\[ \] action_encoder.input_dim is still filled in train.py from dataset dimensions.

\[ \] Parameter count is logged.

\[ \] Commit: config: add shortcut LeWM model graph

# Stage 7 - Add Objective Configurations

**Stage goal**

Make MSE, flow, and shortcut training independently selectable.

| **Files**                            | **Action** |
| ------------------------------------ | ---------- |
| config/train/objective/mse.yaml      | Create     |
| config/train/objective/flow.yaml     | Create     |
| config/train/objective/shortcut.yaml | Create     |

\# mse.yaml  
name: mse  
<br/>\# flow.yaml  
name: flow  
k_max: 8  
target_space: latent  
self_fraction: 0.0  
bootstrap_start_steps: 0  
bootstrap_detach_encoder: true  
alternate_batches: false  
<br/>\# shortcut.yaml  
name: shortcut  
k_max: 8  
target_space: latent  
self_fraction: 0.25  
bootstrap_start_steps: 5000  
bootstrap_detach_encoder: true  
alternate_batches: false

\[ \] objective=mse composes with model=lewm.

\[ \] objective=flow composes with model=shortcut_lewm.

\[ \] objective=shortcut composes with model=shortcut_lewm.

\[ \] Commit: config: add dynamics objectives

# Stage 8 - Update Training Config Composition

**Stage goal**

Add objective selection to the global train defaults without changing current LeWM dimensions or optimization defaults.

| **Files**              | **Action** |
| ---------------------- | ---------- |
| config/train/lewm.yaml | Modify     |

defaults:  
\- \_self_  
\- launcher: local  
\- data: pusht  
\- model: lewm  
\- objective: mse

- Keep img_size=224 for full experiments.
- Keep embed_dim=192, history_size=3, num_preds=1.
- Keep batch size 128, AdamW lr 5e-5, weight decay 1e-3, max_epochs 100, and SIGReg weight 0.09 for matched full runs.

\[ \] Run Hydra config print/resolve for all three variants.

\[ \] Commit: config: compose objective selection

# Stage 9 - Add Training Objective Dispatch

**Stage goal**

Keep the original MSE forward exact and add a separate flow/shortcut forward function.

| **Files**   | **Action**     |
| ----------- | -------------- |
| train.py    | Modify         |
| shortcut.py | Import helpers |

## Baseline branch

- Rename lejepa_forward to lewm_forward only if useful.
- Do not change its calculations or alignment.
- Select it when cfg.objective.name == "mse".

## Flow/shortcut shared preparation

batch\["action"\] = torch.nan_to_num(batch\["action"\], 0.0)  
output = self.model.encode(batch)  
emb = output\["emb"\]  
act_emb = output\["act_emb"\]  
assert cfg.num_preds == 1  
ctx_emb = emb\[:, :cfg.history_size\]  
ctx_act = act_emb\[:, :cfg.history_size\]  
target = emb\[:, 1:cfg.history_size + 1\]  
context = self.model.predict_context(ctx_emb, ctx_act)  
<br/>if cfg.objective.target_space == "latent":  
flow_target = target  
elif cfg.objective.target_space == "delta":  
flow_target = target - ctx_emb  
else:  
raise ValueError(cfg.objective.target_space)

## Flow-only branch

sample = sample_finest_flow_batch(flow_target, cfg.objective.k_max)  
pred_target = self.model.flow_predict(  
sample\["x_t"\], context, sample\["signal_idx"\], sample\["step_idx"\]  
)  
dynamics_loss = flow_xpred_loss(pred_target, flow_target, sample\["weight"\])

## Shortcut branch

- Before bootstrap_start_steps: train all examples with empirical finest flow.
- After bootstrap start: first B_emp rows use empirical flow; last B_self rows use bootstrap consistency.
- B_self = int(B \* self_fraction); B_emp = B - B_self.
- For bootstrap rows, detach target latents and recompute context from detached visual embeddings.
- Do not detach actions or flow-head/context-network parameters.

ctx_emb_self = ctx_emb\[B_emp:\].detach()  
ctx_act_self = ctx_act\[B_emp:\]  
flow_target_self = flow_target\[B_emp:\].detach()  
context_self = self.model.predict_context(ctx_emb_self, ctx_act_self)  
\# Construct x_t on sampled shortcut grid and call shortcut_bootstrap_loss.

## Alternating batches for physical batch size 1

if cfg.objective.alternate_batches and self.global_step >= cfg.objective.bootstrap_start_steps:  
use_shortcut = (self.global_step % 4 == 0) # approximately 25% bootstrap  
else:  
use_shortcut = False

**Local-only behavior**

alternate_batches is required for the 4 GB RTX 3050 Ti comparison because physical batch size 1 cannot be split into empirical and bootstrap rows. Keep split-batch training for full experiments.

## Total loss and logs

sigreg_loss = self.sigreg(emb.transpose(0, 1))  
loss = dynamics_loss + cfg.loss.sigreg.weight \* sigreg_loss

\[ \] Log loss, dynamics_loss, sigreg_loss.

\[ \] For shortcut: log flow_loss, shortcut_loss, and shortcut update ratio.

\[ \] Select forward_fn before constructing spt.Module.

\[ \] Run baseline regression tests after the change.

\[ \] Commit: feat: add flow and shortcut training dispatch

# Stage 10 - Add Evaluation Sampling Configuration

**Stage goal**

Expose NFE and planning-noise seed through Hydra.

| **Files**                               | **Action**                                       |
| --------------------------------------- | ------------------------------------------------ |
| config/eval/sampling/deterministic.yaml | Create                                           |
| config/eval/sampling/shortcut.yaml      | Create                                           |
| config/eval/\*.yaml                     | Compose sampling config or add equivalent fields |

\# deterministic.yaml and shortcut.yaml  
nfe: 1  
num_model_samples: 1  
seed: \${seed}

\[ \] Do not change task horizon, budget, goal offset, or CEM defaults.

\[ \] Baseline model ignores sampling config safely.

\[ \] Shortcut model can override sampling.nfe=1,2,4,8.

\[ \] Commit: config: add evaluation sampling controls

# Stage 11 - Integrate Sampling into eval.py

**Stage goal**

Configure flow-capable checkpoints without altering the existing planner construction.

| **Files** | **Action**       |
| --------- | ---------------- |
| eval.py   | Modify minimally |

model = swm.wm.utils.load_pretrained(cfg.policy)  
model = model.to("cuda").eval()  
model.requires_grad_(False)  
<br/>if hasattr(model, "configure_sampling"):  
model.configure_sampling(  
nfe=cfg.sampling.nfe,  
num_model_samples=cfg.sampling.num_model_samples,  
seed=cfg.sampling.seed,  
)

- Leave hydra.utils.instantiate(cfg.solver, model=model) unchanged.
- Leave WorldModelPolicy unchanged.
- Retain the existing text result log; optionally add JSONL later.

\[ \] Baseline evaluation still starts.

\[ \] Shortcut checkpoint starts for NFE 1,2,4,8.

\[ \] Commit: feat: configure shortcut sampling during evaluation

# Stage 12 - Complete Required Tests

**Stage goal**

Catch schedule, gradient, shape, call-count, and planning-noise errors before training.

| **Files**               | **Action** |
| ----------------------- | ---------- |
| tests/test_baseline.py  | Complete   |
| tests/test_flow_head.py | Complete   |
| tests/test_shortcut.py  | Complete   |

| **Test**             | **Pass condition**                                     |
| -------------------- | ------------------------------------------------------ |
| Schedule             | k_max=8 supports NFE 1,2,4,8; tau+d never exceeds 1    |
| Call count           | NFE 1 calls flow head once; NFE 4 calls it four times  |
| Finite flow loss     | Loss finite and backward succeeds                      |
| Finite shortcut loss | Loss finite and flow/context gradients exist           |
| Encoder detach       | Shortcut-only loss does not update visual encoder      |
| Cost shape           | B=2,S=5 returns (2,5)                                  |
| Common noise         | Repeated get_cost is identical for fixed seed          |
| Context cost         | ARPredictor called once per generated environment step |
| Save/load            | Checkpoint round trip preserves outputs                |
| Baseline regression  | Original MSE path remains numerically stable           |

pytest -q tests/test_baseline.py tests/test_flow_head.py tests/test_shortcut.py

\[ \] All CPU tests pass.

\[ \] GPU integration test passes if CUDA is available.

\[ \] Commit: test: cover shortcut flow training and planning

# Stage 13 - Run One-Epoch Smoke Training

**Stage goal**

Verify all three regimes train, save, reload, and enter evaluation before long runs.

| **Files**            | **Action**   |
| -------------------- | ------------ |
| No new file required | Run commands |

## Baseline

python train.py data=pusht model=lewm objective=mse \\  
trainer.max_epochs=1 output_model_name=smoke_lewm

## Flow

python train.py data=pusht model=shortcut_lewm objective=flow \\  
objective.k_max=4 trainer.max_epochs=1 output_model_name=smoke_flow

## Shortcut

python train.py data=pusht model=shortcut_lewm objective=shortcut \\  
objective.k_max=4 objective.bootstrap_start_steps=0 \\  
trainer.max_epochs=1 output_model_name=smoke_shortcut

\[ \] No NaNs or CUDA errors.

\[ \] Every model saves a checkpoint.

\[ \] Every checkpoint reloads.

\[ \] Shortcut checkpoint runs NFE 1,2,4.

\[ \] Commit: chore: document smoke-test commands

# Stage 13A - RTX 3050 Ti Apples-to-Apples Diagnostic

**Stage goal**

Compare LeWM, Flow-LeWM, and Shortcut-LeWM under identical reduced-scale conditions on 4 GB VRAM.

| **Files**                   | **Action**                                        |
| --------------------------- | ------------------------------------------------- |
| config/train/rtx3050ti.yaml | Create                                            |
| train.py                    | Add deterministic subset limiter and timing hooks |
| benchmark_rtx3050ti.py      | Create                                            |
| eval_dynamics.py            | Later used for common offline metrics             |

**Interpretation limit**

This is an engineering and small-scale scientific diagnostic. It cannot establish full PushT or cross-task superiority.

## Matched settings

| **Setting**           | **Value for all models**          |
| --------------------- | --------------------------------- |
| GPU                   | Single RTX 3050 Ti Laptop, 4 GB   |
| Precision             | 16-mixed                          |
| Image size            | 112 x 112                         |
| Latent dimension      | 192                               |
| History               | 3                                 |
| Physical batch        | 1                                 |
| Gradient accumulation | 8                                 |
| Effective batch       | 8                                 |
| Train subset          | Same deterministic 800 examples   |
| Validation subset     | Same deterministic 160 examples   |
| Optimizer             | AdamW, lr 5e-5, weight decay 1e-3 |
| Optimizer updates     | 100                               |
| Seed                  | 3072                              |
| Flow k_max            | 4                                 |

## Local config

defaults:  
\- lewm  
\- \_self_  
<br/>seed: 3072  
img_size: 112  
embed_dim: 192  
history_size: 3  
num_preds: 1  
<br/>wandb:  
enabled: false  
<br/>trainer:  
devices: 1  
accelerator: gpu  
precision: 16-mixed  
gradient_clip_val: 1.0  
max_steps: 100  
max_epochs: -1  
accumulate_grad_batches: 8  
num_sanity_val_steps: 0  
log_every_n_steps: 1  
<br/>loader:  
batch_size: 1  
num_workers: 0  
persistent_workers: false  
pin_memory: false  
<br/>debug_subset:  
enabled: true  
train_examples: 800  
val_examples: 160

## Deterministic subset helper

def limit_dataset(dataset, count: int, seed: int):  
count = min(count, len(dataset))  
g = torch.Generator().manual_seed(seed)  
indices = torch.randperm(len(dataset), generator=g)\[:count\].tolist()  
return torch.utils.data.Subset(dataset, indices)

- Use seed for train subset and seed+1 for validation subset.
- Do not include model name in the subset seed.
- Save selected indices or their hash in the result log.

## Training commands

\# LeWM  
python train.py --config-name=rtx3050ti data=pusht model=lewm objective=mse \\  
seed=3072 trainer.max_steps=100 trainer.accumulate_grad_batches=8 \\  
output_model_name=rtx3050ti_lewm  
<br/>\# Flow-LeWM  
python train.py --config-name=rtx3050ti data=pusht model=shortcut_lewm objective=flow \\  
objective.k_max=4 objective.target_space=latent seed=3072 \\  
trainer.max_steps=100 trainer.accumulate_grad_batches=8 \\  
output_model_name=rtx3050ti_flow  
<br/>\# Shortcut-LeWM  
python train.py --config-name=rtx3050ti data=pusht model=shortcut_lewm objective=shortcut \\  
objective.k_max=4 objective.target_space=latent \\  
objective.bootstrap_start_steps=10 objective.alternate_batches=true \\  
seed=3072 trainer.max_steps=100 trainer.accumulate_grad_batches=8 \\  
output_model_name=rtx3050ti_shortcut

## Timing and memory instrumentation

torch.cuda.reset_peak_memory_stats()  
torch.cuda.synchronize()  
start = time.perf_counter()  
\# measured operation  
torch.cuda.synchronize()  
elapsed = time.perf_counter() - start  
peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024\*\*3  
peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024\*\*3

\[ \] Record GPU model, driver, CUDA version, idle VRAM, and laptop power limit if visible.

\[ \] Exclude first five microbatches from throughput summaries.

\[ \] Record median and p95 microbatch time.

\[ \] Record optimizer-step time, updates/minute, and examples/second.

\[ \] Record peak allocated and reserved VRAM.

## Common offline comparison

| **Model/evaluation**     | **NFE** | **Noise**        |
| ------------------------ | ------- | ---------------- |
| LeWM                     | 1       | Deterministic    |
| Flow-LeWM full reference | 4       | Fixed seed 12345 |
| Shortcut-LeWM full       | 4       | Same fixed noise |
| Shortcut-LeWM few-step   | 2       | Same fixed noise |
| Shortcut-LeWM one-step   | 1       | Same fixed noise |

- Evaluate the same 64 validation sequences.
- Use rollout horizons 1, 3, and 5.
- Report latent MSE and cosine similarity.
- Use one fixed stochastic sample; do not use best-of-N.

## Inference benchmark

- B=1, history=3, latent=192, rollout horizon=5.
- 20 warmup iterations and 100 timed iterations.
- Report transition milliseconds, five-step rollout milliseconds, transitions/second, and peak inference VRAM.
- Use a forward hook to prove ARPredictor runs once per generated environment step.

## Reduced CEM diagnostic

| **Setting**      | **Value** |
| ---------------- | --------- |
| Episodes         | 5         |
| Horizon          | 2         |
| Receding horizon | 1         |
| Candidates       | 8         |
| CEM iterations   | 3         |
| Top-k            | 2         |
| Model samples    | 1         |

python eval.py --config-name=pusht.yaml policy=&lt;checkpoint&gt; \\  
eval.num_eval=5 plan_config.horizon=2 plan_config.receding_horizon=1 \\  
solver.num_samples=8 solver.n_steps=3 solver.topk=2 \\  
sampling.nfe=&lt;1|2|4&gt; sampling.seed=12345

## Protocol A and B

| **Protocol**        | **Matched resource**                     | **Question**                                  |
| ------------------- | ---------------------------------------- | --------------------------------------------- |
| A: Equal updates    | 100 optimizer updates, effective batch 8 | Quality with equal data/optimization exposure |
| B: Equal wall clock | Optional 30-minute run per model         | Quality under equal laptop compute budget     |

**Fairness rule**

Flow-LeWM must be evaluated at NFE 4. Shortcut-LeWM must be evaluated from the same checkpoint at NFE 4, 2, and 1. This separates the value of flow dynamics from the quality of shortcut approximation.

\[ \] All three models complete 100 optimizer updates.

\[ \] Same subset, seed, effective batch, and optimizer settings are confirmed.

\[ \] No run exceeds available 4 GB VRAM.

\[ \] Training time and peak VRAM are recorded.

\[ \] One-, three-, and five-step metrics are recorded.

\[ \] Shortcut checkpoint runs at NFE 1, 2, and 4.

\[ \] Fixed-noise results reproduce across repeated runs.

\[ \] Conditioning-shuffle ratios are reported.

\[ \] Reduced CEM completes for every variant.

\[ \] Results are labelled local diagnostics, not benchmark results.

# Stage 14 - Add Offline Dynamics Evaluation

**Stage goal**

Evaluate all models using a common latent-prediction metric independent of their training losses.

| **Files**        | **Action** |
| ---------------- | ---------- |
| eval_dynamics.py | Create     |

- Load a checkpoint and held-out dataset sequences.
- Encode true states with that checkpoint's encoder.
- Use recorded action sequences.
- Evaluate one-step and open-loop horizons 1, 5, 10, and 25 when sequence length permits.
- For the local 3050 Ti run, use horizons 1, 3, and 5.

| **Metric**        | **Notes**                                                       |
| ----------------- | --------------------------------------------------------------- |
| Latent MSE        | Common primary local metric                                     |
| Cosine similarity | Scale-insensitive complement                                    |
| Sample-mean MSE   | Later multi-sample stochastic evaluation                        |
| Best-of-M MSE     | Diagnostic only; never primary comparison to deterministic LeWM |
| Energy score      | Optional full-study distributional metric                       |

\[ \] Fixed data order and fixed noise seed.

\[ \] Save JSON/CSV results.

\[ \] Commit: feat: add offline latent dynamics evaluation

# Stage 15 - Add Conditioning Diagnostics

**Stage goal**

Detect whether flow models use actions and visual context rather than merely matching the SIGReg latent marginal.

| **Files**        | **Action** |
| ---------------- | ---------- |
| eval_dynamics.py | Extend     |

| **Condition**    | **Input change**                      |
| ---------------- | ------------------------------------- |
| Normal           | Correct latent history and actions    |
| Shuffled action  | Shuffle actions across batch          |
| Shuffled context | Shuffle latent histories across batch |

action_shuffle_ratio = shuffled_action_loss / normal_loss  
context_shuffle_ratio = shuffled_context_loss / normal_loss

- Ratios greater than one indicate conditioning matters.
- If both remain near one, print a warning that the flow may be reproducing the latent marginal.
- If conditioning collapse is observed, run Flow-LeWM with target_space=delta before tuning shortcut forcing.

\[ \] Correct, action-shuffled, and context-shuffled metrics saved.

\[ \] Commit: eval: add conditioning dependence diagnostics

# Stage 16 - Run the Full PushT Pilot

**Stage goal**

Determine whether full flow works and whether shortcut forcing approximates it before running all environments.

| **Files**                | **Action**         |
| ------------------------ | ------------------ |
| No code changes expected | Train and evaluate |

| **Run** | **Model**     | **Objective**     | **Evaluation** |
| ------- | ------------- | ----------------- | -------------- |
| A       | LeWM          | MSE               | Deterministic  |
| B       | Flow-LeWM     | Flow, k_max=8     | NFE 8          |
| C       | Shortcut-LeWM | Shortcut, k_max=8 | NFE 8,4,2,1    |

- Use seed 3072, 100 epochs, batch 128, same encoder and SIGReg settings.
- Do not tune hyperparameters before the first matched pilot finishes.
- Record control metric, offline dynamics, conditioning ratios, planning runtime, and VRAM.

\[ \] Base, flow, and shortcut checkpoints trained.

\[ \] PushT evaluation completed.

\[ \] Decision tree in Stage 18 applied.

# Stage 17 - Run the Shortcut NFE Sweep

**Stage goal**

Measure the quality/efficiency frontier from a single Shortcut-LeWM checkpoint.

| **Files**                | **Action** |
| ------------------------ | ---------- |
| No code changes expected | Evaluate   |

python eval.py --config-name=pusht.yaml policy=&lt;shortcut_checkpoint&gt; sampling.nfe=8  
python eval.py --config-name=pusht.yaml policy=&lt;shortcut_checkpoint&gt; sampling.nfe=4  
python eval.py --config-name=pusht.yaml policy=&lt;shortcut_checkpoint&gt; sampling.nfe=2  
python eval.py --config-name=pusht.yaml policy=&lt;shortcut_checkpoint&gt; sampling.nfe=1

\[ \] Same checkpoint and evaluation start states for every NFE.

\[ \] Control metric, get_cost latency, full evaluation time, and peak memory recorded.

\[ \] Flow-LeWM NFE 8 retained as full-flow reference.

# Stage 18 - Use Failure-Directed Decision Tree

**Stage goal**

Change only the component implicated by diagnostics.

| **Files**     | **Action**               |
| ------------- | ------------------------ |
| No fixed file | Apply targeted follow-up |

| **Observed result**                            | **Interpretation**               | **Next action**                                                    |
| ---------------------------------------------- | -------------------------------- | ------------------------------------------------------------------ |
| Flow NFE 8 bad; shuffle ratios near 1          | Direct-latent Gaussian shortcut  | Run target_space=delta; do not tune shortcut yet                   |
| Flow good; Shortcut NFE 8 bad                  | Bootstrap training failure       | Sweep self_fraction 0.125/0.25/0.5 and bootstrap warmup            |
| NFE 8 good; NFE 1 bad                          | Shortcut approximation failure   | Tune shortcut schedule, not CEM                                    |
| Offline good; CEM bad                          | Stochastic planning issue        | Add num_model_samples=4 with common random numbers                 |
| NFE 1 unexpectedly slow                        | Context recomputed per substep   | Profile and ensure one ARPredictor call per environment transition |
| Control improves; probes/conditioning collapse | Representation/dynamics loophole | Prioritize delta target and conditioning tests                     |

# Stage 19 - Run the Full Four-Task Benchmark

**Stage goal**

Evaluate matched models across the existing LeWM tasks after the PushT pilot is stable.

| **Files**                   | **Action**                         |
| --------------------------- | ---------------------------------- |
| Existing train/eval configs | Use without task-parameter changes |

| **Dimension**        | **Values**                     |
| -------------------- | ------------------------------ |
| Tasks                | PushT, TwoRoom, Cube, Reacher  |
| Seeds                | 3072, 3073, 3074               |
| Training variants    | LeWM, Flow-LeWM, Shortcut-LeWM |
| Shortcut evaluations | NFE 8,4,2,1                    |
| Primary planner      | Existing CEM settings          |

- Treat the checked-out repository task configs as the benchmark definition.
- Do not fix unrelated evaluation issues midway. If an evaluation change is made, reevaluate every model.
- Parallelize independent runs across GPUs rather than using multi-GPU training unless needed.

\[ \] All 3 seeds complete per task and model.

\[ \] Mean and standard deviation computed.

\[ \] Runtime and memory included alongside control quality.

# Stage 20 - Produce Final Report

**Stage goal**

Separate model-quality, shortcut-quality, and efficiency conclusions.

| **Files**        | **Action**                           |
| ---------------- | ------------------------------------ |
| results/         | Create structured CSV/JSON and plots |
| README or report | Document commands and findings       |

## Primary control table

| **Task** | **LeWM** | **Flow K8** | **Shortcut K8** | **Shortcut K4** | **Shortcut K2** | **Shortcut K1** |
| -------- | -------- | ----------- | --------------- | --------------- | --------------- | --------------- |
| PushT    |          |             |                 |                 |                 |                 |
| TwoRoom  |          |             |                 |                 |                 |                 |
| Cube     |          |             |                 |                 |                 |                 |
| Reacher  |          |             |                 |                 |                 |                 |

## Efficiency table

| **Model** | **NFE** | **Params** | **get_cost ms** | **Decision ms** | **Peak VRAM** |
| --------- | ------- | ---------- | --------------- | --------------- | ------------- |
| LeWM      | 1       |            |                 |                 |               |
| Flow      | 8       |            |                 |                 |               |
| Shortcut  | 8       |            |                 |                 |               |
| Shortcut  | 4       |            |                 |                 |               |
| Shortcut  | 2       |            |                 |                 |               |
| Shortcut  | 1       |            |                 |                 |               |

## RTX 3050 Ti local table

| **Model**   | **Updates** | **Time** | **Peak VRAM** | **1-step MSE** | **5-step MSE** | **Transition ms** |
| ----------- | ----------- | -------- | ------------- | -------------- | -------------- | ----------------- |
| LeWM        | 100         |          |               |                |                |                   |
| Flow K4     | 100         |          |               |                |                |                   |
| Shortcut K4 | 100         |          |               |                |                |                   |
| Shortcut K2 | \-          |          |               |                |                |                   |
| Shortcut K1 | \-          |          |               |                |                |                   |

\[ \] Clearly label single-seed and local diagnostic results.

\[ \] Report negative results without hiding them.

\[ \] State whether flow helps, whether shortcut preserves flow quality, and whether planning speed remains competitive as three separate findings.

# 26\. File Manifest and Final Acceptance Criteria

## Existing files to modify

module.py  
jepa.py  
train.py  
eval.py  
config/train/lewm.yaml  
config/eval/&lt;task configs only for sampling composition, if necessary&gt;

## Existing files to preserve

config/train/model/lewm.yaml  
Original JEPA class behavior  
Original LeWM MSE forward formula  
Existing CEM solver and latent goal criterion

## New files

shortcut.py  
config/train/model/shortcut_lewm.yaml  
config/train/objective/mse.yaml  
config/train/objective/flow.yaml  
config/train/objective/shortcut.yaml  
config/train/rtx3050ti.yaml  
config/eval/sampling/deterministic.yaml  
config/eval/sampling/shortcut.yaml  
eval_dynamics.py  
benchmark_rtx3050ti.py  
tests/test_baseline.py  
tests/test_flow_head.py  
tests/test_shortcut.py

## First implementation milestone

\[ \] Existing LeWM trains and passes numeric regression.

\[ \] ShortcutFlowHead is implemented and tested.

\[ \] shortcut.py contains all flow/shortcut math.

\[ \] Flow-only training runs.

\[ \] Shortcut training runs.

\[ \] NFE 1/2/4/8 inference runs.

\[ \] CEM can call ShortcutJEPA.get_cost.

\[ \] Common-noise determinism test passes.

\[ \] Checkpoint save/load round trip passes.

\[ \] PushT one-epoch smoke train/eval succeeds.

\[ \] RTX 3050 Ti 100-update comparison completes for all three training variants.

## Final scientific acceptance

\[ \] Full Flow-LeWM is compared directly with base LeWM.

\[ \] Shortcut-LeWM full NFE is compared with Flow-LeWM full NFE.

\[ \] The same shortcut checkpoint is evaluated at 8/4/2/1 NFE.

\[ \] Conditioning dependence is measured.

\[ \] Planning speed and peak memory are reported.

\[ \] Three seeds and four tasks are used before broad benchmark claims.

# Working Notes

Update implementation progress here adn any additional important information