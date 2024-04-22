import copy
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from pytorch_lightning import LightningModule
from torchbricks.brick_collection import BrickCollection
from torchbricks.bricks import BrickMetrics, BrickMetricSingle
from torchbricks.model_stage import DEFAULT_MODEL_STAGE_GROUPS


class ModelStage(Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    INFERENCE = "INFERENCE"


class LightningBrickCollection(LightningModule):
    def __init__(
        self,
        path_experiments: Path,
        experiment_name: Optional[str],
        brick_collection: "BrickCollection",
        create_optimizers_func: Callable,
    ):
        super().__init__()

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S")
        self.path_experiment = path_experiments / experiment_name
        self.path_experiment.mkdir(parents=True)

        # self.model_stage_groups = DEFAULT_MODEL_STAGE_GROUPS
        self.bricks = per_stage_brick_collections(brick_collection)

        self.create_optimizers_func = create_optimizers_func

    def _on_epoch_end(self, stage: ModelStage):
        metrics = self.bricks[stage.value].summarize(reset=True)

        def is_single_value(tensor):
            return isinstance(tensor, torch.Tensor) and tensor.ndim == 0

        stage_str = stage.value.lower()
        single_valued_metrics = {f"{stage_str}/{metric_name}": value for metric_name, value in metrics.items() if is_single_value(value)}
        self.log_dict(single_valued_metrics)

    def _step(self, stage: ModelStage, batch, batch_idx: int):
        stage_str = stage.value.lower()
        named_inputs = {"raw": batch[0], "targets": batch[1], "batch_idx": batch_idx}
        named_outputs = self.bricks[stage.value](named_inputs=named_inputs, groups=DEFAULT_MODEL_STAGE_GROUPS[stage.value])
        losses = self.bricks[stage.value].extract_losses(named_outputs=named_outputs)
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


def copy_metric_bricks(brick_collection: BrickCollection) -> BrickCollection:
    brick_collection = dict(brick_collection)
    for brick_name, brick in brick_collection.items():
        if isinstance(brick, BrickCollection):
            brick_collection[brick_name] = copy_metric_bricks(brick)
        elif isinstance(brick, (BrickMetrics, BrickMetricSingle)):
            # brick_metrics = brick_collection.pop(brick_name)
            # brick_metrics.metrics = brick_metrics.metrics.clone()
            brick_collection[brick_name] = copy.deepcopy(brick)

    return brick_collection


def per_stage_brick_collections(brick_collection: BrickCollection) -> Dict[str, BrickCollection]:
    # We need to keep metrics separated for each model stage (train, validation, test), we do that by making a brick collection for each
    # stage. Only the metrics are copied to each stage - other bricks are shared. This is necessary because PyTorch Lightning
    # runs all training steps, all validation steps and then 'on_validation_epoch_end' and 'on_train_epoch_end'.
    # Meaning that all metrics aggregated, summarized and reset in the on_validation_epoch_end and nothing is stored in
    # 'on_train_epoch_end'
    metric_stages = [ModelStage.TRAIN, ModelStage.VALIDATION, ModelStage.TEST]
    return torch.nn.ModuleDict({stage.value: BrickCollection(copy_metric_bricks(brick_collection)) for stage in metric_stages})
