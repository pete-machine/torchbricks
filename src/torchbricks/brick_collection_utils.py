import copy
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import torch
from torch import nn
from typeguard import typechecked

from torchbricks import model_stage
from torchbricks.brick_collection import BrickCollection


@typechecked
class _OnnxExportAdaptor(nn.Module):
    def __init__(self, model: nn.Module, groups: Set[str]) -> None:
        super().__init__()
        self.model = model
        self.groups = groups or model_stage.EXPORT

    def forward(self, named_inputs: Dict[str, Any]):
        named_outputs = self.model.forward(named_inputs=named_inputs, groups=self.groups, return_inputs=False)
        return named_outputs


@typechecked
def export_bricks_as_onnx(
    path_onnx: Path,
    brick_collection: BrickCollection,
    named_inputs: Dict[str, torch.Tensor],
    dynamic_batch_size: bool,
    groups: Optional[Set[str]] = None,
    **onnx_export_kwargs,
):
    groups = groups or model_stage.EXPORT
    outputs = brick_collection(named_inputs=named_inputs, groups=groups, return_inputs=False)
    onnx_exportable = _OnnxExportAdaptor(model=brick_collection, groups=groups)
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
