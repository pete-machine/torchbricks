from pathlib import Path

import onnx
import pytest
import torch
from utils_testing.utils_testing import create_dummy_brick_collection

from torchbricks.brick_collection import BrickCollection
from torchbricks.brick_collection_utils import export_bricks_as_onnx
from torchbricks.brick_tags import Tag


def test_export_onnx_trace(tmp_path: Path):
    num_classes = 3
    brick_collection = create_dummy_brick_collection(num_classes=num_classes, num_backbone_featues=10)
    model = BrickCollection(brick_collection)
    named_inputs = {"raw": torch.zeros((1, 3, 64, 64))}
    tags = {Tag.MODEL}
    named_outputs = model(named_inputs, tags=tags, return_inputs=False)
    # remove_from_outputs = ["stage"] + list(named_inputs)
    expected_input = list(named_inputs)
    expected_outputs = list(named_outputs)

    path_onnx = Path(tmp_path / "model.onnx")

    dynamic_batch_size_configs = [False, True]
    for dynamic_batch_size in dynamic_batch_size_configs:
        export_bricks_as_onnx(
            brick_collection=model, named_inputs=named_inputs, tags=tags, path_onnx=path_onnx, dynamic_batch_size=dynamic_batch_size
        )

        assert path_onnx.exists()
        onnx_model = onnx.load(path_onnx)

        output_names_graph = {output.name for output in onnx_model.graph.output}
        assert set(expected_outputs) == output_names_graph

        input_names_graph = {input.name for input in onnx_model.graph.input}
        assert set(expected_input) == input_names_graph

        onnx.checker.check_model(onnx_model)


@pytest.mark.xfail
def test_export_torch_jit_script(tmp_path: Path):
    num_classes = 3
    brick_collection = create_dummy_brick_collection(num_classes=num_classes, num_backbone_featues=10)

    model = BrickCollection(brick_collection).sub_collection(tags={Tag.MODEL})
    named_inputs = {"raw": torch.zeros((1, 3, 64, 64))}

    named_outputs = model(named_inputs, return_inputs=False)
    # remove_from_outputs = ["stage"] + list(named_inputs)
    list(named_inputs)
    list(named_outputs)

    torch.jit.script(model)
