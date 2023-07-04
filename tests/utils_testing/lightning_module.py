from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from pytorch_lightning import LightningModule
import torch

from torchbricks.bricks import BrickCollection, Phase



class LightningBrickCollection(LightningModule):
    def __init__(self, path_experiments: Path,
                 experiment_name: Optional[str],
                 brick_collection: BrickCollection,
                 create_optimizers_func: Callable):
        super().__init__()

        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y_%m_%d_T_%H_%M_%S')
        self.path_experiment = path_experiments / experiment_name
        self.path_experiment.mkdir(parents=True)

        self.bricks = brick_collection
        self.create_optimizers_func = create_optimizers_func

    def _on_epoch_end(self, phase: Phase):
        metrics = self.bricks.summarize(phase=phase, reset=True)
        def is_single_value_tensor(tensor):
            return isinstance(tensor, torch.Tensor) and tensor.ndim == 0

        single_valued_metrics = {metric_name: value for metric_name, value in metrics.items() if is_single_value_tensor(value)}
        self.log_dict(single_valued_metrics)

    def _step(self, phase: Phase, batch, batch_idx: int):
        named_inputs = {'raw': batch[0], 'targets': batch[1], 'batch_idx': batch_idx}
        named_outputs = self.bricks(phase=phase, named_inputs=named_inputs)
        losses = self.bricks.extract_losses(named_outputs=named_outputs)
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
        return self.create_optimizers_func(self.parameters())
