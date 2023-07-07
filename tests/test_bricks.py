from pathlib import Path
import pytest
from torchbricks import bricks
from torchbricks.bricks import BrickCollection, BrickLoss, BrickTorchMetric, Phase
from torchbricks import custom_metrics
from typing import Dict
import torch
import onnx
from torch import nn
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy
from torchbricks.bricks import export_as_onnx

from utils_testing.utils_testing import assert_equal_dictionaries

def create_brick_collection(num_classes: int, num_backbone_featues: int) -> Dict[str, bricks.BrickInterface]:
    class Preprocessor(nn.Module):
        def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
            return raw_input/2

    class TinyBackbone(nn.Module):
        def __init__(self, n_kernels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_kernels, kernel_size=3, padding=1)

        def forward(self, tensor: torch.Tensor):
            return self.conv1(tensor)


    class Classifier(nn.Module):
        def __init__(self, input_channels: int, num_classes: int):
            super().__init__()
            self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Conv2d(in_channels=input_channels, out_channels=num_classes, kernel_size=1)

        def forward(self, features: torch.Tensor):
            logits = torch.squeeze(self.classifier(self.global_average_pooling(features)))
            return logits


    metric_collection = torchmetrics.MetricCollection({
        'MeanAccuracy': MulticlassAccuracy(num_classes=num_classes, average='macro'),
        'Accuracy': MulticlassAccuracy(num_classes=num_classes, average='micro'),
        'ConfMat': torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes),
        'Concatenate': custom_metrics.ConcatenatePredictionAndTarget(compute_on_cpu=True)
    })
    brick_collections = {
        'preprocessor': bricks.BrickNotTrainable(Preprocessor(), input_names=['raw'], output_names=['preprocessed']),
        'backbone': bricks.BrickTrainable(TinyBackbone(n_kernels=num_backbone_featues), input_names=['preprocessed'],
                                                       output_names=['features']),
        'classifier': bricks.BrickTrainable(Classifier(input_channels=num_backbone_featues, num_classes=num_classes),
                                            input_names=['features'],
                                            output_names=['predictions']),
        'loss': bricks.BrickLoss(nn.CrossEntropyLoss(), input_names=['predictions', 'labels'], output_names=['ce_loss']),
        'metrics': bricks.BrickTorchMetric(metric_collection, input_names=['predictions', 'labels'])
    }
    return brick_collections


def test_brick_collection():
    num_classes = 10
    expected_forward_named_outputs = {'labels', 'raw', 'phase', 'preprocessed', 'features', 'predictions'}
    expected_named_losses = {'ce_loss'}
    expected_named_metrics = {'train/Accuracy', 'train/Concatenate', 'train/ConfMat', 'train/MeanAccuracy'}
    brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=5)
    model = bricks.BrickCollection(bricks=brick_collection)
    named_inputs = {'labels': torch.tensor(range(num_classes), dtype=torch.float64), 'raw': torch.zeros((3, 24, 24))}

    named_outputs = model(phase=Phase.TRAIN, named_inputs=named_inputs)
    metrics = model.summarize(phase=Phase.TRAIN, reset=True)

    losses = model.extract_losses(named_outputs=named_outputs)
    assert set(metrics) == expected_named_metrics
    assert expected_forward_named_outputs.union(expected_named_losses) == set(named_outputs)
    assert expected_named_losses == set(losses)


def test_brick_collection_no_metrics():
    num_classes = 10
    expected_forward_named_outputs = {'labels', 'raw', 'phase', 'preprocessed', 'features', 'predictions'}
    expected_named_losses = {'ce_loss'}
    expected_named_metrics = {}

    brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=5)
    brick_collection = {name: brick for name, brick in brick_collection.items() if not isinstance(brick, bricks.BrickTorchMetric)}
    model = bricks.BrickCollection(bricks=brick_collection)

    named_inputs = {'labels': torch.tensor(range(num_classes), dtype=torch.float64), 'raw': torch.zeros((3, 24, 24))}
    named_outputs = model(phase=Phase.INFERENCE, named_inputs=named_inputs)
    assert expected_forward_named_outputs == set(named_outputs)

    named_outputs = model(phase=Phase.TRAIN, named_inputs=named_inputs)
    named_outputs = model(phase=Phase.TRAIN, named_inputs=named_inputs)
    metrics = model.summarize(phase=Phase.TRAIN, reset=True)
    losses = model.extract_losses(named_outputs=named_outputs)
    assert metrics == expected_named_metrics
    assert expected_forward_named_outputs.union(expected_named_losses) == set(named_outputs)
    assert expected_named_losses == set(losses)


def test_brick_collection_no_metrics_no_losses():
    num_classes = 10
    expected_forward_named_outputs = {'labels', 'raw', 'phase', 'preprocessed', 'features', 'predictions'}
    expected_named_losses = {}
    expected_named_metrics = {}

    brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=5)
    brick_collection = {name: brick for name, brick in brick_collection.items() if not isinstance(brick, bricks.BrickTorchMetric)}
    brick_collection = {name: brick for name, brick in brick_collection.items() if not isinstance(brick, bricks.BrickLoss)}
    model = bricks.BrickCollection(bricks=brick_collection)

    named_inputs = {'labels': torch.tensor(range(num_classes), dtype=torch.float64), 'raw': torch.zeros((3, 24, 24))}
    named_outputs = model(phase= Phase.INFERENCE, named_inputs=named_inputs)
    assert expected_forward_named_outputs == set(named_outputs)

    model(phase=Phase.TRAIN, named_inputs=named_inputs)
    model(phase=Phase.TRAIN, named_inputs=named_inputs)
    named_outputs = model(phase=Phase.TRAIN, named_inputs=named_inputs)
    metrics = model.summarize(phase=Phase.TRAIN, reset=True)
    losses = model.extract_losses(named_outputs=named_outputs)
    assert metrics == expected_named_metrics
    assert expected_forward_named_outputs.union(expected_named_losses) == set(named_outputs)
    assert expected_named_losses == losses

def test_nested_bricks():

    class PreprocessorHalf(nn.Module):
        def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
            return raw_input/2

    class PreprocessorSquareRoot(nn.Module):
        def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(raw_input)

    def root_bricks():
        return {
            'preprocessor0': bricks.BrickNotTrainable(PreprocessorHalf(), input_names=['in0'], output_names=['out1']),
            'preprocessor1': bricks.BrickNotTrainable(PreprocessorHalf(), input_names=['out1'], output_names=['out2'])
            }

    def nested_bricks():
        return {
            'preprocessor11': bricks.BrickNotTrainable(PreprocessorSquareRoot(), input_names=['out2'], output_names=['out3']),
            'preprocessor12': bricks.BrickNotTrainable(PreprocessorSquareRoot(), input_names=['out3'], output_names=['out4']),
        }

    # Nested bricks using nested brick collections
    nested_brick_collection = root_bricks()
    nested_brick_collection['collection'] = bricks.BrickCollection(nested_bricks())
    brick_collection = bricks.BrickCollection(bricks=nested_brick_collection)

    # Nested bricks using nested dictionary of bricks
    nested_brick_dict = root_bricks()
    nested_brick_dict['collection'] = nested_bricks()
    brick_collection_dict = bricks.BrickCollection(bricks=nested_brick_dict)

    # No nesting of bricks in flat/single level dictionary
    flat_brick_dict = root_bricks()
    flat_brick_dict.update(nested_bricks())
    brick_collection_flat = bricks.BrickCollection(bricks=nested_brick_dict)


    named_inputs = {'in0': torch.tensor(range(10), dtype=torch.float64)}
    outputs0 = brick_collection(named_inputs=named_inputs, phase=Phase.TRAIN)
    outputs1 = brick_collection_dict(named_inputs=named_inputs, phase=Phase.TRAIN)
    outputs2 = brick_collection_flat(named_inputs=named_inputs, phase=Phase.TRAIN)
    assert_equal_dictionaries(outputs0, outputs1)
    assert_equal_dictionaries(outputs1, outputs2)

    expected_outputs = dict(named_inputs)
    expected_outputs['out1'] = expected_outputs['in0']/2
    expected_outputs['out2'] = expected_outputs['out1']/2
    expected_outputs['out3'] = torch.sqrt(expected_outputs['out2'])
    expected_outputs['out4'] = torch.sqrt(expected_outputs['out3'])
    outputs0.pop('phase')
    assert_equal_dictionaries(outputs0, expected_outputs)


    outputs0 = brick_collection(named_inputs=named_inputs, phase=Phase.TRAIN)
    outputs1 = brick_collection_dict(named_inputs=named_inputs, phase=Phase.TRAIN)
    outputs2 = brick_collection_flat(named_inputs=named_inputs, phase=Phase.TRAIN)

    assert_equal_dictionaries(outputs0, outputs1)
    assert_equal_dictionaries(outputs1, outputs2)


def test_brick_torch_metric_single_metric():
    num_classes = 5
    metric_name = 'Accuracy'
    bricks = {
        'accuracy': BrickTorchMetric(MulticlassAccuracy(num_classes=num_classes),
                                     input_names=['logits', 'targets'],
                                     metric_name=metric_name),
        'loss': BrickLoss(model=nn.CrossEntropyLoss(), input_names=['logits', 'targets'], output_names=['loss_ce'])
    }

    model = BrickCollection(bricks)
    batch_logits = torch.rand((1, num_classes))
    phase = Phase.TRAIN
    named_inputs = {'logits': batch_logits, 'targets': torch.ones((1), dtype=torch.int64)}
    model(phase=phase, named_inputs=named_inputs)
    metrics = model.summarize(phase=phase, reset=True)

    assert list(metrics) == [f'{phase.value}/{metric_name}']


def test_brick_torch_metric_single_metric_assert():
    metric_name = None
    with pytest.raises(ValueError, match='Specify `metric_name` when using'):
        BrickTorchMetric(MulticlassAccuracy(num_classes=10), input_names=['logits', 'targets'], metric_name=metric_name)



def test_brick_torch_metric_multiple_metric():
    num_classes = 5
    metric_collection = torchmetrics.MetricCollection({
        'MeanAccuracy': MulticlassAccuracy(num_classes=num_classes, average='macro'),
        'Accuracy': MulticlassAccuracy(num_classes=num_classes, average='micro'),
        'ConfMat': torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes),
        'Concatenate': custom_metrics.ConcatenatePredictionAndTarget(compute_on_cpu=True)
    })

    metric_name = None # None is allowed for metric collection.
    bricks = {
        'metrics': BrickTorchMetric(metric_collection, input_names=['logits', 'targets'], metric_name=metric_name),
        'loss': BrickLoss(model=nn.CrossEntropyLoss(), input_names=['logits', 'targets'], output_names=['loss_ce'])
    }

    model = BrickCollection(bricks)
    batch_logits = torch.rand((1, num_classes))
    phase = Phase.TRAIN
    named_inputs = {'logits': batch_logits, 'targets': torch.ones((1), dtype=torch.int64)}
    model(named_inputs=named_inputs, phase=phase)
    metrics = model.summarize(phase=phase, reset=True)

    expected_metrics = {f'{phase.value}/{name}' for name in metric_collection}
    assert set(metrics) == expected_metrics

def test_save_and_load_of_brick_collection(tmp_path: Path):
    brick_collection = create_brick_collection(num_classes=3, num_backbone_featues=10)
    model = BrickCollection(brick_collection)
    path_model = tmp_path / 'test_model.pt'

    # Trainable parameters are saved
    torch.save(model.state_dict(), path_model)

    # Trainable parameters are loaded
    model.load_state_dict(torch.load(path_model))

def iterate_phases():
    num_classes = 3
    brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=10)
    model = BrickCollection(brick_collection)

    named_inputs = {'labels': torch.tensor(range(num_classes), dtype=torch.float64), 'raw': torch.zeros((3, 24, 24))}
    for phase in Phase:
        model(named_inputs=named_inputs, phase=phase)
        model.on_step(named_inputs=named_inputs, phase=phase, batch_idx=0)
        model.summarize(phase, reset=True)


@pytest.mark.slow
@pytest.mark.parametrize('phase', [Phase.TRAIN, Phase.INFERENCE])
def test_compile(phase: Phase):
    num_classes = 3
    brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=10)
    model = BrickCollection(brick_collection)

    named_inputs = {'labels': torch.tensor(range(num_classes), dtype=torch.float64), 'raw': torch.zeros((1, 3, 24, 24))}
    forward_expected = model(named_inputs=named_inputs, phase=phase)

    model_compiled = torch.compile(model)
    forward_actual = model_compiled(named_inputs=named_inputs, phase=phase)

    assert_equal_dictionaries(forward_expected, forward_actual, is_close=True)


def test_export_onnx_trace(tmp_path: Path):
    num_classes = 3
    brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=10)
    model = BrickCollection(brick_collection)
    named_inputs = {'raw': torch.zeros((1, 3, 64, 64))}

    phase = Phase.EXPORT
    named_outputs = model(named_inputs, phase=phase, return_inputs=False)
    # remove_from_outputs = ["phase"] + list(named_inputs)
    expected_input = list(named_inputs)
    expected_outputs = list(named_outputs)

    path_onnx = Path(tmp_path / 'model.onnx')

    dynamic_batch_size_configs = [False, True]
    for dynamic_batch_size in dynamic_batch_size_configs:
        export_as_onnx(brick_collection=model, named_inputs=named_inputs, path_onnx=path_onnx, phase=phase,
                       dynamic_batch_size=dynamic_batch_size)

        assert path_onnx.exists()
        onnx_model = onnx.load(path_onnx)

        output_names_graph = set(output.name for output in onnx_model.graph.output)
        assert set(expected_outputs) == output_names_graph

        input_names_graph = set(input.name for input in onnx_model.graph.input)
        assert set(expected_input) == input_names_graph

        onnx.checker.check_model(onnx_model)


# def test_export_jit_script():
#     num_classes = 3
#     brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=10)
#     model = BrickCollection(brick_collection)
#     # named_inputs = {"named_inputs": {'raw': torch.zeros((3, 24, 24))}, "phase": Phase.INFERENCE}
#     named_inputs = {'raw': torch.zeros((3, 24, 24))}
#     # model_scripted = torch.jit.script(model)  # Export to TorchScript
#     # model_scripted.save('model_scripted.pt')  # Save

#     kwargs = {"named_inputs": named_inputs, "phase": Phase.INFERENCE}
#     outputs = model(**kwargs)

#     class OnnxExportable(nn.Module):
#         def __init__(self, model: nn.Module, phase: Phase) -> None:
#             super().__init__()
#             self.model = model
#             self.phase = phase

#         def forward(self, named_inputs):
#             return self.model.forward(named_inputs=named_inputs, phase=self.phase)
#     onnx_exportable = OnnxExportable(model=model, phase=Phase.INFERENCE)
#     torch.onnx.export(model=onnx_exportable, args=tuple(named_inputs.values()),
#                       f="model.onnx", verbose=True,
#                       input_names=list(named_inputs),
#                       output_names=list(outputs))
