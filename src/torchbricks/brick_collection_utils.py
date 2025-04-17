import copy
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import torch
from torch import nn
from typeguard import typechecked

from torchbricks import model_stage
from torchbricks.brick_collection import BrickCollection
from torchbricks.bricks import BrickMetrics, BrickMetricSingle
from torchbricks.model_stage import ModelStage


@typechecked
class _OnnxExportAdaptor(nn.Module):
    def __init__(self, model: nn.Module, tags: Set[str]) -> None:
        super().__init__()
        self.model = model.sub_collection(tags=tags)

    def forward(self, named_inputs: Dict[str, Any]):
        named_outputs = self.model.forward(named_inputs=named_inputs, return_inputs=False)
        return named_outputs


@typechecked
def export_bricks_as_onnx(
    path_onnx: Path,
    brick_collection: BrickCollection,
    named_inputs: Dict[str, torch.Tensor],
    dynamic_batch_size: bool,
    tags: Optional[Set[str]] = None,
    **onnx_export_kwargs,
):
    tags = tags or model_stage.EXPORT
    outputs = brick_collection(named_inputs=named_inputs, tags=tags, return_inputs=False)
    onnx_exportable = _OnnxExportAdaptor(model=brick_collection, tags=tags)
    output_names = list(outputs)
    input_names = list(named_inputs)

    if dynamic_batch_size:
        if "dynamic_axes" in onnx_export_kwargs:
            raise ValueError("Setting both 'dynamic_batch_size==True' and defining 'dynamic_axes' in 'onnx_export_kwargs' is not allowed. ")
        io_names = input_names + output_names
        dynamic_axes = {io_name: {0: "batch_size"} for io_name in io_names}
        onnx_export_kwargs["dynamic_axes"] = dynamic_axes

    torch.onnx.export(
        model=onnx_exportable,
        args=({"named_inputs": named_inputs},),
        f=str(path_onnx),
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        **onnx_export_kwargs,
    )


def filter_brick_types(brick_collection: BrickCollection, types: Tuple) -> Dict[str, Any]:
    brick_collection = copy.copy(brick_collection)
    metrics = {}
    for brick_name, brick in brick_collection.items():
        if isinstance(brick, BrickCollection):
            metrics[brick_name] = filter_brick_types(brick, types=types)
        elif isinstance(brick, types):
            metrics[brick_name] = brick_collection.pop(brick_name)
    return metrics


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
    return torch.nn.ModuleDict({stage.name: BrickCollection(copy_metric_bricks(brick_collection)) for stage in metric_stages})
