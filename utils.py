import json
from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class GPUMetricsCallback(Callback):
    def __init__(self, output_path):
        super().__init__()
        self.output_path = Path(output_path)

    def on_fit_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_fit_end(self, trainer, pl_module):
        if not torch.cuda.is_available():
            return
        metrics = {
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        }
        self.output_path.write_text(json.dumps(metrics, indent=2))


class SaveCkptCallback(Callback):
    """Save best (by val loss) and latest checkpoints."""

    def __init__(self, run_name, cfg, val_loss_key="validate/loss"):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.val_loss_key = val_loss_key
        self.best_val_loss = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if not trainer.is_global_zero:
            return
        val_loss = trainer.callback_metrics.get(self.val_loss_key)
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save(pl_module.model, "best")

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._save(pl_module.model, "latest")

    def _save(self, model, tag):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_{tag}.pt',
        )