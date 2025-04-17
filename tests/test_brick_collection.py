import copy
import shutil
import textwrap
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from torch import nn
from utils_testing.utils_testing import assert_equal_dictionaries, create_dummy_brick_collection, is_equal_model_parameters

import torchbricks.brick_collection
from torchbricks import bricks, model_stage
from torchbricks.brick_collection import BrickCollection
from torchbricks.bricks import BrickModule, Tag


def test_brick_collection():
    num_classes = 10
    brick_collection = create_dummy_brick_collection(num_classes=num_classes, num_backbone_featues=5)
    expected_forward_named_outputs = {"labels", "raw", "preprocessed", "features", "predictions"}
    expected_named_losses = {"ce_loss"}
    expected_named_metrics = set(brick_collection["head"]["metrics"].metrics)

    model = torchbricks.brick_collection.BrickCollection(bricks=brick_collection)

    named_inputs = {"labels": torch.tensor(range(num_classes), dtype=torch.float64), "raw": torch.zeros((3, 24, 24))}

    named_outputs = model(named_inputs=named_inputs)
    metrics = model.summarize(reset=True)

    losses = model.extract_losses(named_outputs=named_outputs)
    assert set(metrics) == expected_named_metrics
    assert expected_forward_named_outputs.union(expected_named_losses) == set(named_outputs)
    assert expected_named_losses == set(losses)


def test_save_and_load_brick_collection(tmp_path: Path):
    # Consider splitting this in multiple tests
    num_classes = 10
    brick_collection = create_dummy_brick_collection(num_classes=num_classes, num_backbone_featues=5)
    model_checkpoint = torchbricks.brick_collection.BrickCollection(bricks=copy.deepcopy(brick_collection))
    with torch.no_grad():
        model_checkpoint["backbone"].model.conv1.weight[0, 0] = 100
        model_checkpoint["head"]["classifier"].model.classifier.weight[0, 0] = 10
    path_model_checkpoint_file = tmp_path / "vanilla.pt"
    path_model_checkpoint_folder = tmp_path / "model"
    model_checkpoint.save_bricks(path_model_checkpoint_folder)
    torch.save(model_checkpoint.state_dict(), path_model_checkpoint_file)

    ## Test: Pytorch save and load ##
    model = torchbricks.brick_collection.BrickCollection(bricks=copy.deepcopy(brick_collection))
    assert not is_equal_model_parameters(model_checkpoint, model)
    torch.save(model_checkpoint.state_dict(), path_model_checkpoint_file)

    model.load_state_dict(torch.load(path_model_checkpoint_file))
    assert is_equal_model_parameters(model_checkpoint, model)

    ## Test: torchbricks save and load ##
    model = torchbricks.brick_collection.BrickCollection(bricks=copy.deepcopy(brick_collection))
    assert not is_equal_model_parameters(model_checkpoint, model)

    model.load_bricks(path_model_checkpoint_folder)
    assert is_equal_model_parameters(model_checkpoint, model)

    # Test: Warnings when model file doesn't exist
    path_head_classifier = path_model_checkpoint_folder / "head" / "classifier.pt"
    assert path_head_classifier.exists()
    path_head_classifier.unlink()
    with pytest.warns(UserWarning, match="Brick 'head/classifier' has no matching weight file"):
        model.load_bricks(path_model_checkpoint_folder)

    path_head = path_model_checkpoint_folder / "head"
    assert path_head.exists()
    shutil.rmtree(path_head)
    with pytest.warns(UserWarning, match="Brick 'head/classifier' has no matching weight file"):
        model.load_bricks(path_model_checkpoint_folder)

    # Test: Check Error being raised when model file is not a directory
    path_model = path_model_checkpoint_folder / "backbone.pt"
    assert path_model.exists()
    with pytest.raises(NotADirectoryError):
        model.load_bricks(path_model)

    # Test: Check Error being raised when model folder path doesn't exist
    path_model = Path("buggy_path")
    with pytest.raises(FileNotFoundError):
        model.load_bricks(path_model)


def test_brick_collection_no_metrics():
    num_classes = 10
    expected_forward_named_outputs = {"labels", "raw", "preprocessed", "features", "predictions"}
    expected_named_losses = {"ce_loss"}
    expected_named_metrics = {}

    brick_collection = create_dummy_brick_collection(num_classes=num_classes, num_backbone_featues=5)
    brick_collection["head"].pop("metrics")
    model = torchbricks.brick_collection.BrickCollection(bricks=brick_collection)

    named_inputs = {"labels": torch.tensor(range(num_classes), dtype=torch.float64), "raw": torch.zeros((3, 24, 24))}
    named_outputs = model(named_inputs=named_inputs, tags=model_stage.INFERENCE)
    assert expected_forward_named_outputs == set(named_outputs)

    named_outputs = model(named_inputs=named_inputs)
    named_outputs = model(named_inputs=named_inputs)
    metrics = model.summarize(reset=True)
    losses = model.extract_losses(named_outputs=named_outputs)
    assert metrics == expected_named_metrics
    assert expected_forward_named_outputs.union(expected_named_losses) == set(named_outputs)
    assert expected_named_losses == set(losses)


def test_brick_collection_no_metrics_no_losses():
    num_classes = 10
    expected_forward_named_outputs = {"labels", "raw", "preprocessed", "features", "predictions"}
    expected_named_losses = {}
    expected_named_metrics = {}

    brick_collection = create_dummy_brick_collection(num_classes=num_classes, num_backbone_featues=5)
    brick_collection["head"].pop("metrics")
    brick_collection["head"].pop("loss")
    model = torchbricks.brick_collection.BrickCollection(bricks=brick_collection)

    named_inputs = {"labels": torch.tensor(range(num_classes), dtype=torch.float64), "raw": torch.zeros((3, 24, 24))}
    named_outputs = model(named_inputs=named_inputs, tags=model_stage.INFERENCE)
    assert expected_forward_named_outputs == set(named_outputs)

    model(named_inputs=named_inputs)
    model(named_inputs=named_inputs)
    named_outputs = model(named_inputs=named_inputs)
    metrics = model.summarize(reset=True)
    losses = model.extract_losses(named_outputs=named_outputs)
    assert metrics == expected_named_metrics
    assert expected_forward_named_outputs.union(expected_named_losses) == set(named_outputs)
    assert expected_named_losses == losses


def test_brick_collection_to_dict():
    brick_collection = create_dummy_brick_collection(num_classes=10, num_backbone_featues=5)
    model = torchbricks.brick_collection.BrickCollection(bricks=brick_collection)

    model_as_dict = model.to_dict()
    assert isinstance(model_as_dict, dict)
    assert isinstance(model_as_dict["head"], dict)

    model_as_dict["backbone"].input_names = ["changed"]
    assert model_as_dict["backbone"].input_names == model["backbone"].input_names


def test_brick_collection_sub_collection():
    brick_collection = create_dummy_brick_collection(num_classes=10, num_backbone_featues=5)
    model = torchbricks.brick_collection.BrickCollection(bricks=brick_collection)

    model_sub = model.sub_collection()
    assert model_sub.keys() == model.keys()
    assert model_sub["head"].keys() == model["head"].keys()
    assert set(model_sub["head"].keys()) == {"classifier", "loss", "metrics"}

    model_sub = model.sub_collection(tags={Tag.MODEL})
    assert set(model_sub["head"].keys()) == {"classifier"}


def test_nested_bricks():
    class PreprocessorHalf(nn.Module):
        def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
            return raw_input / 2

    class PreprocessorSquareRoot(nn.Module):
        def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(raw_input)

    def root_bricks():
        return {
            "preprocessor0": bricks.BrickNotTrainable(PreprocessorHalf(), input_names=["in0"], output_names=["out1"]),
            "preprocessor1": bricks.BrickNotTrainable(PreprocessorHalf(), input_names=["out1"], output_names=["out2"]),
        }

    def nested_bricks():
        return {
            "preprocessor11": bricks.BrickNotTrainable(PreprocessorSquareRoot(), input_names=["out2"], output_names=["out3"]),
            "preprocessor12": bricks.BrickNotTrainable(PreprocessorSquareRoot(), input_names=["out3"], output_names=["out4"]),
        }

    # Nested bricks using nested brick collections
    nested_brick_collection = root_bricks()
    nested_brick_collection["collection"] = torchbricks.brick_collection.BrickCollection(nested_bricks())
    brick_collection = torchbricks.brick_collection.BrickCollection(bricks=nested_brick_collection)

    # Nested bricks using nested dictionary of bricks
    nested_brick_dict = root_bricks()
    nested_brick_dict["collection"] = nested_bricks()
    brick_collection_dict = torchbricks.brick_collection.BrickCollection(bricks=nested_brick_dict)

    # No nesting of bricks in flat/single level dictionary
    flat_brick_dict = root_bricks()
    flat_brick_dict.update(nested_bricks())
    brick_collection_flat = torchbricks.brick_collection.BrickCollection(bricks=nested_brick_dict)

    named_inputs = {"in0": torch.tensor(range(10), dtype=torch.float64)}
    outputs0 = brick_collection(named_inputs=named_inputs)
    outputs1 = brick_collection_dict(named_inputs=named_inputs)
    outputs2 = brick_collection_flat(named_inputs=named_inputs)
    assert_equal_dictionaries(outputs0, outputs1)
    assert_equal_dictionaries(outputs1, outputs2)

    expected_outputs = dict(named_inputs)
    expected_outputs["out1"] = expected_outputs["in0"] / 2
    expected_outputs["out2"] = expected_outputs["out1"] / 2
    expected_outputs["out3"] = torch.sqrt(expected_outputs["out2"])
    expected_outputs["out4"] = torch.sqrt(expected_outputs["out3"])
    assert_equal_dictionaries(outputs0, expected_outputs)

    outputs0 = brick_collection(named_inputs=named_inputs)
    outputs1 = brick_collection_dict(named_inputs=named_inputs)
    outputs2 = brick_collection_flat(named_inputs=named_inputs)

    assert_equal_dictionaries(outputs0, outputs1)
    assert_equal_dictionaries(outputs1, outputs2)


def test_brick_collection_print():
    num_classes = 10
    brick_collection_as_dict = create_dummy_brick_collection(num_classes=num_classes, num_backbone_featues=5)
    brick_collection = BrickCollection(brick_collection_as_dict)

    expected_str = textwrap.dedent("""\
        BrickCollection(
          (preprocessor): BrickNotTrainable(Preprocessor, input_names=['raw'], output_names=['preprocessed'], tags={'MODEL'})
          (backbone): BrickTrainable(TinyBackbone, input_names=['preprocessed'], output_names=['features'], tags={'MODEL'})
          (head): BrickCollection(
            (classifier): BrickTrainable(Classifier, input_names=['features'], output_names=['predictions'], tags={'MODEL'})
            (loss): BrickLoss(CrossEntropyLoss, input_names=['predictions', 'labels'], output_names=['ce_loss'], tags={'LOSS'})
            (metrics): BrickMetrics(['Accuracy', 'Concatenate', 'ConfMat', 'MeanAccuracy'], input_names=['predictions', 'labels'], output_names=[], tags={'METRIC'})
          )
        )""")  # noqa: E501
    assert brick_collection.__str__() == expected_str


def test_resolve_relative_names():
    brick_collection_as_dict = {
        "preprocessor": BrickModule(model=nn.Identity(), input_names=["raw"], output_names=["processed"]),
        "backbone": BrickModule(model=nn.Identity(), input_names=["processed"], output_names=["embeddings"]),
        "head0": {
            "classifier": BrickModule(model=nn.Identity(), input_names=["../embeddings"], output_names=["./predictions"]),
            "loss": BrickModule(model=nn.Identity(), input_names=["./predictions"], output_names=["./loss"]),
        },
        "head1": {
            "classifier": BrickModule(model=nn.Identity(), input_names=["embeddings"], output_names=["./predictions"]),
            "loss": BrickModule(model=nn.Identity(), input_names=["./predictions"], output_names=["./loss"]),
            "head1_nested": {
                "classifier": BrickModule(
                    model=nn.Identity(),
                    input_names=["../../embeddings", "../predictions", "../../head0/predictions"],
                    output_names=["./predictions"],
                ),
                "loss": BrickModule(model=nn.Identity(), input_names=["./predictions"], output_names=["./loss"]),
            },
        },
    }

    model = BrickCollection(brick_collection_as_dict)
    assert model["head0"]["classifier"].input_names == ["embeddings"]
    assert model["head0"]["classifier"].output_names == ["head0/predictions"]
    assert model["head0"]["loss"].input_names == ["head0/predictions"]
    assert model["head0"]["loss"].output_names == ["head0/loss"]

    assert model["head1"]["head1_nested"]["classifier"].input_names == ["embeddings", "head1/predictions", "head0/predictions"]
    assert model["head1"]["head1_nested"]["classifier"].output_names == ["head1/head1_nested/predictions"]

    assert model["head1"]["head1_nested"]["loss"].input_names == ["head1/head1_nested/predictions"]
    assert model["head1"]["head1_nested"]["loss"].output_names == ["head1/head1_nested/loss"]


def test_resolve_relative_names_dict():
    class SomeDummyLoss(nn.Module):
        def forward(self, tensor: torch.Tensor, named_data: Dict[str, Any]) -> torch.Tensor:
            assert set(named_data.keys()) == {"raw", "processed", "embeddings", "head0/predictions"}
            return tensor

    brick_collection_as_dict = {
        "preprocessor": BrickModule(model=nn.Identity(), input_names=["raw"], output_names=["processed"]),
        "backbone": BrickModule(model=nn.Identity(), input_names=["processed"], output_names=["embeddings"]),
        "head0": {
            "classifier": BrickModule(model=nn.Identity(), input_names=["../embeddings"], output_names=["./predictions"]),
            "loss": BrickModule(
                model=SomeDummyLoss(), input_names={"tensor": "./predictions", "named_data": "__all__"}, output_names=["./loss"]
            ),
        },
    }

    model = BrickCollection(brick_collection_as_dict)
    assert model["head0"]["classifier"].input_names == ["embeddings"]
    assert model["head0"]["classifier"].output_names == ["head0/predictions"]
    assert model["head0"]["loss"].input_names == {"tensor": "head0/predictions", "named_data": "__all__"}
    assert model["head0"]["loss"].output_names == ["head0/loss"]

    model(named_inputs={"raw": torch.rand((2, 3, 10, 20))})


def test_resolve_relative_names_errors():
    bricks = {
        "preprocessor": BrickModule(model=nn.Identity(), input_names=["raw"], output_names=["processed"]),
        "backbone": BrickModule(model=nn.Identity(), input_names=["processed"], output_names=["embeddings"]),
        "head0": {
            "classifier": BrickModule(model=nn.Identity(), input_names=["../../embeddings"], output_names=["./predictions"]),
            "loss": BrickModule(model=nn.Identity(), input_names=["./predictions"], output_names=["./loss"]),
        },
    }
    with pytest.raises(ValueError, match="Failed to resolve input name. Unable to resolve"):
        BrickCollection(bricks)


def test_no_inputs_or_outputs():
    class NoInputsNoOutputs(torch.nn.Module):
        def forward(self) -> None:
            return None

    bricks = {
        "preprocessor": BrickModule(model=NoInputsNoOutputs(), input_names=[], output_names=[]),
    }

    brick_collection = BrickCollection(bricks)
    brick_collection(named_inputs={"raw": torch.rand((2, 3, 100, 200))}, tags=model_stage.INFERENCE)


def test_input_names_all():
    dict_bricks = create_dummy_brick_collection(num_classes=3, num_backbone_featues=5)

    class VisualizePredictions(torch.nn.Module):
        def forward(self, named_inputs: Dict[str, Any]):
            assert len(named_inputs) == 4
            return torch.concatenate((named_inputs["raw"], named_inputs["preprocessed"]))

    dict_bricks["Visualize"] = bricks.BrickNotTrainable(VisualizePredictions(), input_names=["__all__"], output_names=["visualized"])
    brick_collection = BrickCollection(dict_bricks)
    brick_collection(named_inputs={"raw": torch.rand((2, 3, 100, 200))}, tags=model_stage.INFERENCE)


def test_save_and_load_of_brick_collection(tmp_path: Path):
    brick_collection = create_dummy_brick_collection(num_classes=3, num_backbone_featues=10)
    model = BrickCollection(brick_collection)
    path_model = tmp_path / "test_model.pt"

    # Trainable parameters are saved
    torch.save(model.state_dict(), path_model)

    # Trainable parameters are loaded
    model.load_state_dict(torch.load(path_model))


@pytest.mark.slow
@pytest.mark.parametrize("tags", [None, model_stage.INFERENCE])
def test_compile(tags):
    torch._dynamo.config.suppress_errors = True  # TODO: Remove this line later when possible
    num_classes = 3
    brick_collection = create_dummy_brick_collection(num_classes=num_classes, num_backbone_featues=10)
    model = BrickCollection(brick_collection)

    named_inputs = {"labels": torch.tensor(range(num_classes), dtype=torch.float64), "raw": torch.zeros((1, 3, 24, 24))}
    forward_expected = model(named_inputs=named_inputs, tags=tags)

    model_compiled = torch.compile(model)
    forward_actual = model_compiled(named_inputs=named_inputs, tags=tags)

    assert_equal_dictionaries(forward_expected, forward_actual, is_close=True)
