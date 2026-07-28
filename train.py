import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from shortcut import (
    sample_finest_flow_batch,
    flow_xpred_loss,
    sample_shortcut_grid,
    shortcut_bootstrap_loss,
)
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def limit_dataset(dataset, count, seed):
    count = min(count, len(dataset))
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:count].tolist()
    return torch.utils.data.Subset(dataset, indices)


def lewm_forward(self, batch, stage, cfg):
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def flow_forward(self, batch, stage, cfg):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :cfg.history_size]
    ctx_act = act_emb[:, :cfg.history_size]
    target = emb[:, 1:cfg.history_size + 1]
    context = self.model.predict_context(ctx_emb, ctx_act)
    if cfg.objective.target_space == "latent":
        flow_target = target
    elif cfg.objective.target_space == "delta":
        flow_target = target - ctx_emb
    else:
        raise ValueError(cfg.objective.target_space)
    sample = sample_finest_flow_batch(flow_target, cfg.objective.k_max)
    pred_target = self.model.flow_predict(
        sample["x_t"], context, sample["signal_idx"], sample["step_idx"]
    )
    dynamics_loss = flow_xpred_loss(pred_target, flow_target, sample["weight"])
    lambd = cfg.loss.sigreg.weight
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = dynamics_loss + lambd * output["sigreg_loss"]
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def shortcut_forward(self, batch, stage, cfg):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    B = emb.size(0)
    ctx_emb = emb[:, :cfg.history_size]
    ctx_act = act_emb[:, :cfg.history_size]
    target = emb[:, 1:cfg.history_size + 1]
    context = self.model.predict_context(ctx_emb, ctx_act)
    if cfg.objective.target_space == "latent":
        flow_target = target
    elif cfg.objective.target_space == "delta":
        flow_target = target - ctx_emb
    else:
        raise ValueError(cfg.objective.target_space)
    use_alternate = cfg.objective.alternate_batches
    bootstrap_started = self.global_step >= cfg.objective.bootstrap_start_steps
    if use_alternate and bootstrap_started:
        use_shortcut = (self.global_step % 4 == 0)
    else:
        use_shortcut = False
    if use_shortcut:
        sample = sample_finest_flow_batch(flow_target, cfg.objective.k_max)
        pred_target = self.model.flow_predict(
            sample["x_t"], context, sample["signal_idx"], sample["step_idx"]
        )
        flow_loss = flow_xpred_loss(pred_target, flow_target, sample["weight"])
        output["flow_loss"] = flow_loss.detach()
        skw = sample_shortcut_grid(flow_target.shape[:2], cfg.objective.k_max, flow_target.device)
        shortcut_tgt = flow_target
        shortcut_ctx = ctx_emb.detach()
        shortcut_act = ctx_act
        shortcut_flow = flow_target.detach()
        shortcut_context = self.model.predict_context(shortcut_ctx, shortcut_act)
        noise = torch.randn_like(shortcut_flow)
        tau = skw["tau"][..., None]
        x_t = (1.0 - tau) * noise + tau * shortcut_flow
        shortcut_loss = shortcut_bootstrap_loss(
            self.model.flow_head, x_t, shortcut_context,
            skw["signal_idx"], skw["step_idx"],
            skw["tau"], skw["d"], cfg.objective.k_max,
        )
        output["shortcut_loss"] = shortcut_loss.detach()
        dynamics_loss = 0.5 * flow_loss + 0.5 * shortcut_loss
        output["shortcut_ratio"] = torch.tensor(0.5, device=flow_loss.device)
    elif cfg.objective.self_fraction > 0.0 and bootstrap_started and not use_alternate:
        B_self = max(1, int(B * cfg.objective.self_fraction))
        B_emp = B - B_self
        sample = sample_finest_flow_batch(flow_target[:B_emp], cfg.objective.k_max)
        pred_target = self.model.flow_predict(
            sample["x_t"], context[:B_emp], sample["signal_idx"], sample["step_idx"]
        )
        flow_loss = flow_xpred_loss(pred_target, flow_target[:B_emp], sample["weight"])
        output["flow_loss"] = flow_loss.detach()
        shortcut_ctx = ctx_emb[B_emp:].detach()
        shortcut_act = ctx_act[B_emp:]
        shortcut_flow = flow_target[B_emp:].detach()
        shortcut_context = self.model.predict_context(shortcut_ctx, shortcut_act)
        noise = torch.randn_like(shortcut_flow)
        skw = sample_shortcut_grid(shortcut_flow.shape[:2], cfg.objective.k_max, shortcut_flow.device)
        tau = skw["tau"][..., None]
        x_t = (1.0 - tau) * noise + tau * shortcut_flow
        shortcut_loss = shortcut_bootstrap_loss(
            self.model.flow_head, x_t, shortcut_context,
            skw["signal_idx"], skw["step_idx"],
            skw["tau"], skw["d"], cfg.objective.k_max,
        )
        output["shortcut_loss"] = shortcut_loss.detach()
        dynamics_loss = (B_emp * flow_loss + B_self * shortcut_loss) / B
        output["shortcut_ratio"] = torch.tensor(B_self / B, device=flow_loss.device)
    else:
        sample = sample_finest_flow_batch(flow_target, cfg.objective.k_max)
        pred_target = self.model.flow_predict(
            sample["x_t"], context, sample["signal_idx"], sample["step_idx"]
        )
        dynamics_loss = flow_xpred_loss(pred_target, flow_target, sample["weight"])
        output["flow_loss"] = dynamics_loss.detach()
        output["shortcut_loss"] = torch.tensor(0.0, device=dynamics_loss.device)
        output["shortcut_ratio"] = torch.tensor(0.0, device=dynamics_loss.device)
    lambd = cfg.loss.sigreg.weight
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = dynamics_loss + lambd * output["sigreg_loss"]
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    if cfg.get("debug_subset", {}).get("enabled", False):
        train_set = limit_dataset(train_set, cfg.debug_subset.train_examples, cfg.seed)
        val_set = limit_dataset(val_set, cfg.debug_subset.val_examples, cfg.seed + 1)

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    objective_name = cfg.get("objective", {}).get("name", "mse")
    if objective_name == "mse":
        forward_fn = lewm_forward
    elif objective_name == "flow":
        forward_fn = flow_forward
    elif objective_name == "shortcut":
        forward_fn = shortcut_forward
    else:
        forward_fn = lewm_forward

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(forward_fn, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
