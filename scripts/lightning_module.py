from pathlib import Path
from typing import Callable
from pytorch_lightning import LightningModule
import torch

from torchbricks.bricks import BrickCollection, Phase



class LightningBrickCollection(LightningModule):
    def __init__(self, path_experiments: Path,
                 experiment_name: str,
                 brick_collection: BrickCollection,
                 create_optimizer_func: Callable,
                 create_lr_scheduler_func: Callable = None):
        super().__init__()

        self.path_experiment = path_experiments / experiment_name
        self.path_experiment.mkdir()

        self.bricks = brick_collection
        self.create_optimizer_func = create_optimizer_func
        self.create_lr_scheduler_func = create_lr_scheduler_func


    def _on_epoch_end(self, phase: Phase):
        metrics = self.bricks.summarize(phase=phase, reset=True)
        def is_single_value_tensor(tensor):
            return isinstance(tensor, torch.Tensor) and tensor.ndim == 0

        single_valued_metrics = {metric_name: value for metric_name, value in metrics.items() if is_single_value_tensor(value)}
        self.log_dict(single_valued_metrics)

    def _step(self, phase: Phase, batch, batch_idx: int):
        named_inputs = {'raw': batch[0], 'targets': batch[1]}
        _, losses = self.bricks.on_step(phase=phase, named_inputs=named_inputs, batch_idx=batch_idx)
        loss = 0
        for loss_name, loss_value in losses.items():
            self.log(f'{phase.value}/{loss_name}', loss_value)
            loss = loss + loss_value
        self.log(f'{phase.value}/total_loss', loss)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self._step(phase=Phase.TRAIN, batch=batch, batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx: int):
        self._step(phase=Phase.VALIDATION, batch=batch, batch_idx=batch_idx)

    def test_step(self, batch, batch_idx):
        self._step(phase=Phase.TEST, batch=batch, batch_idx=batch_idx)

    def on_train_epoch_end(self):
        self._on_epoch_end(phase=Phase.TRAIN)

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(phase=Phase.VALIDATION)

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(phase=Phase.TEST)

    def configure_optimizers(self):
        optimizer = self.create_optimizer_func(self.parameters())
        optimizers = {
            'optimizer': optimizer,
            'lr_scheduler': self.create_lr_scheduler_func(optimizer)
            }
        return optimizers
