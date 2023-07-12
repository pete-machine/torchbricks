from pathlib import Path
from typing import Optional

import pytest
import torch
import torchmetrics
from torch import nn
from torchbricks import bricks, custom_metrics
from torchbricks.bricks import BrickCollection, BrickLoss, BrickMetrics, BrickMetricSingle, Stage
from torchmetrics.classification import MulticlassAccuracy
from utils_testing.utils_testing import assert_equal_dictionaries, create_brick_collection


def test_brick_collection():
    num_classes = 10
    brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=5)
    expected_forward_named_outputs = {'labels', 'raw', 'stage', 'preprocessed', 'features', 'predictions'}
    expected_named_losses = {'ce_loss'}
    expected_named_metrics = set(brick_collection['metrics'].metrics[Stage.TRAIN.name])

    model = bricks.BrickCollection(bricks=brick_collection)
    named_inputs = {'labels': torch.tensor(range(num_classes), dtype=torch.float64), 'raw': torch.zeros((3, 24, 24))}

    named_outputs = model(stage=Stage.TRAIN, named_inputs=named_inputs)
    metrics = model.summarize(stage=Stage.TRAIN, reset=True)

    losses = model.extract_losses(named_outputs=named_outputs)
    assert set(metrics) == expected_named_metrics
    assert expected_forward_named_outputs.union(expected_named_losses) == set(named_outputs)
    assert expected_named_losses == set(losses)


def test_brick_collection_no_metrics():
    num_classes = 10
    expected_forward_named_outputs = {'labels', 'raw', 'stage', 'preprocessed', 'features', 'predictions'}
    expected_named_losses = {'ce_loss'}
    expected_named_metrics = {}

    brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=5)
    brick_collection = {name: brick for name, brick in brick_collection.items() if not isinstance(brick, bricks.BrickMetrics)}
    model = bricks.BrickCollection(bricks=brick_collection)

    named_inputs = {'labels': torch.tensor(range(num_classes), dtype=torch.float64), 'raw': torch.zeros((3, 24, 24))}
    named_outputs = model(stage=Stage.INFERENCE, named_inputs=named_inputs)
    assert expected_forward_named_outputs == set(named_outputs)

    named_outputs = model(stage=Stage.TRAIN, named_inputs=named_inputs)
    named_outputs = model(stage=Stage.TRAIN, named_inputs=named_inputs)
    metrics = model.summarize(stage=Stage.TRAIN, reset=True)
    losses = model.extract_losses(named_outputs=named_outputs)
    assert metrics == expected_named_metrics
    assert expected_forward_named_outputs.union(expected_named_losses) == set(named_outputs)
    assert expected_named_losses == set(losses)


def test_brick_collection_no_metrics_no_losses():
    num_classes = 10
    expected_forward_named_outputs = {'labels', 'raw', 'stage', 'preprocessed', 'features', 'predictions'}
    expected_named_losses = {}
    expected_named_metrics = {}

    brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=5)
    brick_collection = {name: brick for name, brick in brick_collection.items() if not isinstance(brick, bricks.BrickMetrics)}
    brick_collection = {name: brick for name, brick in brick_collection.items() if not isinstance(brick, bricks.BrickLoss)}
    model = bricks.BrickCollection(bricks=brick_collection)

    named_inputs = {'labels': torch.tensor(range(num_classes), dtype=torch.float64), 'raw': torch.zeros((3, 24, 24))}
    named_outputs = model(stage= Stage.INFERENCE, named_inputs=named_inputs)
    assert expected_forward_named_outputs == set(named_outputs)

    model(stage=Stage.TRAIN, named_inputs=named_inputs)
    model(stage=Stage.TRAIN, named_inputs=named_inputs)
    named_outputs = model(stage=Stage.TRAIN, named_inputs=named_inputs)
    metrics = model.summarize(stage=Stage.TRAIN, reset=True)
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
    outputs0 = brick_collection(named_inputs=named_inputs, stage=Stage.TRAIN)
    outputs1 = brick_collection_dict(named_inputs=named_inputs, stage=Stage.TRAIN)
    outputs2 = brick_collection_flat(named_inputs=named_inputs, stage=Stage.TRAIN)
    assert_equal_dictionaries(outputs0, outputs1)
    assert_equal_dictionaries(outputs1, outputs2)

    expected_outputs = dict(named_inputs)
    expected_outputs['out1'] = expected_outputs['in0']/2
    expected_outputs['out2'] = expected_outputs['out1']/2
    expected_outputs['out3'] = torch.sqrt(expected_outputs['out2'])
    expected_outputs['out4'] = torch.sqrt(expected_outputs['out3'])
    outputs0.pop('stage')
    assert_equal_dictionaries(outputs0, expected_outputs)


    outputs0 = brick_collection(named_inputs=named_inputs, stage=Stage.TRAIN)
    outputs1 = brick_collection_dict(named_inputs=named_inputs, stage=Stage.TRAIN)
    outputs2 = brick_collection_flat(named_inputs=named_inputs, stage=Stage.TRAIN)

    assert_equal_dictionaries(outputs0, outputs1)
    assert_equal_dictionaries(outputs1, outputs2)


@pytest.mark.parametrize('metric_name', ('accuracy', None))
def test_brick_torch_metric_single_metric(metric_name: Optional[str]):
    num_classes = 5
    bricks = {
        'accuracy': BrickMetricSingle(MulticlassAccuracy(num_classes=num_classes), input_names=['logits', 'targets'],
                                      metric_name=metric_name),
        'loss': BrickLoss(model=nn.CrossEntropyLoss(), input_names=['logits', 'targets'], output_names=['loss_ce'])
    }

    model = BrickCollection(bricks)
    batch_logits = torch.rand((1, num_classes))
    stage = Stage.TRAIN
    named_inputs = {'logits': batch_logits, 'targets': torch.ones((1), dtype=torch.int64)}
    model(named_inputs=named_inputs, stage=stage)
    metrics = model.summarize(stage=stage, reset=True)
    expected_metric_name = metric_name or 'MulticlassAccuracy'

    assert list(metrics) == [expected_metric_name]


@pytest.mark.parametrize('return_metrics', [False, True])
def test_brick_torch_metric_multiple_metric(return_metrics: bool):
    num_classes = 5
    metric_collection = torchmetrics.MetricCollection({
        'MeanAccuracy': MulticlassAccuracy(num_classes=num_classes, average='macro'),
        'Accuracy': MulticlassAccuracy(num_classes=num_classes, average='micro'),
        'ConfMat': torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes),
        'Concatenate': custom_metrics.ConcatenatePredictionAndTarget(compute_on_cpu=True)
    })

    bricks = {
        'metrics': BrickMetrics(metric_collection, input_names=['logits', 'targets'], return_metrics=return_metrics),
        'loss': BrickLoss(model=nn.CrossEntropyLoss(), input_names=['logits', 'targets'], output_names=['loss_ce'])
    }

    model = BrickCollection(bricks)
    batch_logits = torch.rand((1, num_classes))
    stage = Stage.TRAIN
    named_inputs = {'logits': batch_logits, 'targets': torch.ones((1), dtype=torch.int64)}
    named_outputs = model(named_inputs=named_inputs, stage=stage)

    expected_outputs = {'logits', 'targets', 'stage', 'loss_ce'}
    if return_metrics:
        expected_outputs = expected_outputs.union(set(metric_collection))

    assert set(named_outputs) == expected_outputs
    metrics = model.summarize(stage=stage, reset=True)
    expected_metrics = set(metric_collection)
    assert set(metrics) == expected_metrics

def test_save_and_load_of_brick_collection(tmp_path: Path):
    brick_collection = create_brick_collection(num_classes=3, num_backbone_featues=10)
    model = BrickCollection(brick_collection)
    path_model = tmp_path / 'test_model.pt'

    # Trainable parameters are saved
    torch.save(model.state_dict(), path_model)

    # Trainable parameters are loaded
    model.load_state_dict(torch.load(path_model))

def iterate_stages():
    num_classes = 3
    brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=10)
    model = BrickCollection(brick_collection)

    named_inputs = {'labels': torch.tensor(range(num_classes), dtype=torch.float64), 'raw': torch.zeros((3, 24, 24))}
    for stage in Stage:
        model(named_inputs=named_inputs, stage=stage)
        model.on_step(named_inputs=named_inputs, stage=stage, batch_idx=0)
        model.summarize(stage, reset=True)


@pytest.mark.slow
@pytest.mark.parametrize('stage', [Stage.TRAIN, Stage.INFERENCE])
def test_compile(stage: Stage):
    num_classes = 3
    brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=10)
    model = BrickCollection(brick_collection)

    named_inputs = {'labels': torch.tensor(range(num_classes), dtype=torch.float64), 'raw': torch.zeros((1, 3, 24, 24))}
    forward_expected = model(named_inputs=named_inputs, stage=stage)

    model_compiled = torch.compile(model)
    forward_actual = model_compiled(named_inputs=named_inputs, stage=stage)

    assert_equal_dictionaries(forward_expected, forward_actual, is_close=True)
