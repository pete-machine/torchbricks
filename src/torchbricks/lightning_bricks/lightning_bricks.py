from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch
from pytorch_lightning import LightningModule

from torchbricks.brick_collection import BrickCollection
from torchbricks.brick_collection_utils import per_stage_brick_collections
from torchbricks.brick_tags import DEFAULT_MODEL_STAGE_TAGS, ModelStage


class LightningBricks(LightningModule):
    def __init__(
        self,
        path_experiments: Path,
        experiment_name: Optional[str],
        brick_collection: BrickCollection,
        create_optimizers_func: Callable,
    ):
        super().__init__()

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S")
        self.path_experiment = path_experiments / experiment_name
        self.path_experiment.mkdir(parents=True)

        self.model_stage_tags = DEFAULT_MODEL_STAGE_TAGS
        self.bricks = per_stage_brick_collections(brick_collection)

        self.create_optimizers_func = create_optimizers_func

    def _on_epoch_end(self, stage: ModelStage):
        metrics = self.bricks[stage.name].summarize(reset=True)

        def is_single_value(tensor):
            return isinstance(tensor, torch.Tensor) and tensor.ndim == 0

        stage_str = stage.value.lower()
        single_valued_metrics = {f"{stage_str}/{metric_name}": value for metric_name, value in metrics.items() if is_single_value(value)}
        self.log_dict(single_valued_metrics)

    def _step(self, stage: ModelStage, batch, batch_idx: int):
        stage_str = stage.value.lower()
        # named_inputs = {"raw": batch[0], "targets": batch[1], "batch_idx": batch_idx}
        named_outputs = self.bricks[stage.name](named_inputs=batch, tags=self.model_stage_tags[stage.name])
        losses = self.bricks[stage.name].extract_losses(named_outputs=named_outputs)
        loss = 0
        for loss_name, loss_value in losses.items():
            self.log(f"{stage_str}/{loss_name}", loss_value)
            loss = loss + loss_value
        self.log(f"{stage_str}/total_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx: int):
        with torch.inference_mode():
            return self._step(stage=ModelStage.INFERENCE, batch=batch, batch_idx=batch_idx)

    def training_step(self, batch, batch_idx: int):
        return self._step(stage=ModelStage.TRAIN, batch=batch, batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx: int):
        with torch.inference_mode():
            self._step(stage=ModelStage.VALIDATION, batch=batch, batch_idx=batch_idx)

    def test_step(self, batch, batch_idx):
        with torch.inference_mode():
            self._step(stage=ModelStage.TEST, batch=batch, batch_idx=batch_idx)

    def on_train_epoch_end(self):
        self._on_epoch_end(stage=ModelStage.TRAIN)

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(stage=ModelStage.VALIDATION)

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(stage=ModelStage.TEST)

    def configure_optimizers(self):
        return self.create_optimizers_func(self.parameters())
