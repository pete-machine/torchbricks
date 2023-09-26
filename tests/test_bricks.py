import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import torch
import torchmetrics
from torch import nn
from torchbricks import bricks, custom_metrics
from torchbricks.bricks import BrickCollection, BrickLoss, BrickMetrics, BrickMetricSingle, BrickModule, BrickUnbatched, Stage
from torchmetrics.classification import MulticlassAccuracy
from typeguard import typechecked
from utils_testing.utils_testing import assert_equal_dictionaries, create_brick_collection


def test_brick_collection():
    num_classes = 10
    brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=5)
    expected_forward_named_outputs = {'labels', 'raw', 'stage', 'preprocessed', 'features', 'predictions'}
    expected_named_losses = {'ce_loss'}
    expected_named_metrics = set(brick_collection['head']['metrics'].metrics[Stage.TRAIN.name])

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
    brick_collection['head'].pop('metrics')
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
    brick_collection['head'].pop('metrics')
    brick_collection['head'].pop('loss')
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


def test_brick_collection_print():
    num_classes = 10
    brick_collection_as_dict = create_brick_collection(num_classes=num_classes, num_backbone_featues=5)
    brick_collection = BrickCollection(brick_collection_as_dict)

    expected_str = textwrap.dedent('''\
        BrickCollection(
          (preprocessor): BrickNotTrainable(Preprocessor, input_names=['raw'], output_names=['preprocessed'], alive_stages=['TRAIN', 'VALIDATION', 'TEST', 'INFERENCE', 'EXPORT'])
          (backbone): BrickTrainable(TinyBackbone, input_names=['preprocessed'], output_names=['features'], alive_stages=['TRAIN', 'VALIDATION', 'TEST', 'INFERENCE', 'EXPORT'])
          (head): BrickCollection(
            (classifier): BrickTrainable(Classifier, input_names=['features'], output_names=['predictions'], alive_stages=['TRAIN', 'VALIDATION', 'TEST', 'INFERENCE', 'EXPORT'])
            (loss): BrickLoss(CrossEntropyLoss, input_names=['predictions', 'labels'], output_names=['ce_loss'], alive_stages=['TRAIN', 'TEST', 'VALIDATION'])
            (metrics): BrickMetrics(['Accuracy', 'Concatenate', 'ConfMat', 'MeanAccuracy'], input_names=['predictions', 'labels'], output_names=[], alive_stages=['TRAIN', 'TEST', 'VALIDATION'])
          )
        )''') # noqa: E501
    assert brick_collection.__str__() == expected_str


def test_resolve_relative_names():
    brick_collection_as_dict = {
        'preprocessor': BrickModule(model=nn.Identity(), input_names=['raw'], output_names=['processed']),
        'backbone': BrickModule(model=nn.Identity(), input_names=['processed'], output_names=['embeddings']),
        'head0': {
            'classifier': BrickModule(model=nn.Identity(), input_names=['../embeddings'], output_names=['./predictions']),
            'loss': BrickModule(model=nn.Identity(), input_names=['./predictions'], output_names=['./loss']),
        },
        'head1': {
            'classifier': BrickModule(model=nn.Identity(), input_names=['embeddings'], output_names=['./predictions']),
            'loss': BrickModule(model=nn.Identity(), input_names=['./predictions'], output_names=['./loss']),
            'head1_nested':{
                'classifier': BrickModule(model=nn.Identity(), input_names=['../../embeddings',
                                                                            '../predictions',
                                                                            '../../head0/predictions'],
                                          output_names=['./predictions']),
                'loss': BrickModule(model=nn.Identity(), input_names=['./predictions'], output_names=['./loss']),
            }
        }
    }

    model = BrickCollection(brick_collection_as_dict)
    assert model['head0']['classifier'].input_names == ['embeddings']
    assert model['head0']['classifier'].output_names == ['head0/predictions']
    assert model['head0']['loss'].input_names == ['head0/predictions']
    assert model['head0']['loss'].output_names == ['head0/loss']

    assert model['head1']['head1_nested']['classifier'].input_names == ['embeddings', 'head1/predictions', 'head0/predictions']
    assert model['head1']['head1_nested']['classifier'].output_names == ['head1/head1_nested/predictions']

    assert model['head1']['head1_nested']['loss'].input_names == ['head1/head1_nested/predictions']
    assert model['head1']['head1_nested']['loss'].output_names == ['head1/head1_nested/loss']

def test_resolve_relative_names_dict():
    class SomeDummyLoss(nn.Module):
        def forward(self, tensor: torch.Tensor, named_data: Dict[str, Any]) -> torch.Tensor:
            assert set(named_data.keys()) == {'stage', 'raw', 'processed', 'embeddings', 'head0/predictions'}
            return tensor

    brick_collection_as_dict = {
        'preprocessor': BrickModule(model=nn.Identity(), input_names=['raw'], output_names=['processed']),
        'backbone': BrickModule(model=nn.Identity(), input_names=['processed'], output_names=['embeddings']),
        'head0': {
            'classifier': BrickModule(model=nn.Identity(), input_names=['../embeddings'], output_names=['./predictions']),
            'loss': BrickModule(model=SomeDummyLoss(), input_names={'tensor': './predictions', 'named_data': '__all__'},
                                     output_names=['./loss']),
        },
    }

    model = BrickCollection(brick_collection_as_dict)
    assert model['head0']['classifier'].input_names == ['embeddings']
    assert model['head0']['classifier'].output_names == ['head0/predictions']
    assert model['head0']['loss'].input_names == {'tensor': 'head0/predictions', 'named_data': '__all__'}
    assert model['head0']['loss'].output_names == ['head0/loss']

    model(named_inputs={'raw': torch.rand((2, 3, 10, 20))}, stage=Stage.TRAIN)


def test_brick_tensor_as_arrays_single_output_name():
    @typechecked
    def draw_function(array: np.ndarray, named_data: Dict[str, Any]) -> np.ndarray:
        assert array.shape == (10, 20, 3)
        assert set(named_data.keys()) == {'stage', 'raw', 'processed', 'embeddings'}
        return array

    brick_collection_as_dict = {
        'preprocessor': BrickModule(model=nn.Identity(), input_names=['raw'], output_names=['processed']),
        'backbone': BrickModule(model=nn.Identity(), input_names=['processed'], output_names=['embeddings']),
        'visualizer': BrickUnbatched(callable=draw_function, input_names=['raw', '__all__'], output_names=['visualized']),
    }

    batch_size = 5
    model = BrickCollection(brick_collection_as_dict)
    outputs = model(named_inputs={'raw': torch.rand((batch_size, 3, 10, 20))}, stage=Stage.INFERENCE)
    assert len(outputs['visualized']) == batch_size
    assert outputs['visualized'][0].shape == (10, 20, 3)
    assert set(outputs) == {'raw', 'processed', 'embeddings', 'visualized', 'stage'}

def test_brick_tensor_as_arrays_two_output_names():
    @typechecked
    def draw_function(array0: np.ndarray, array1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert array0.shape == (10, 20, 3)
        assert array1.shape == (10, 20, 3)
        return array0, array1

    brick_collection_as_dict = {
        'preprocessor': BrickModule(model=nn.Identity(), input_names=['raw'], output_names=['processed']),
        'backbone': BrickModule(model=nn.Identity(), input_names=['processed'], output_names=['embeddings']),
        'visualizer': BrickUnbatched(callable=draw_function, input_names=['raw', 'processed'],
                                     output_names=['visualized0', 'visualized1']),
    }

    batch_size = 5
    model = BrickCollection(brick_collection_as_dict)
    outputs = model(named_inputs={'raw': torch.rand((batch_size, 3, 10, 20))}, stage=Stage.INFERENCE)
    assert len(outputs['visualized0']) == batch_size
    assert len(outputs['visualized1']) == batch_size
    assert outputs['visualized0'][0].shape == (10, 20, 3)
    assert outputs['visualized1'][0].shape == (10, 20, 3)

@pytest.mark.parametrize(['permute_tensors', 'tensors_to_numpy'], [(False, False), (True, False), (True, True), (False, True)])
def test_brick_tensor_as_arrays_two_output_names_params(permute_tensors, tensors_to_numpy):
    @typechecked
    def draw_function(array0, arrays: np.ndarray):
        assert arrays.shape == (10, 20, 3)  # List of arrays should be untouched by the function for all cases
        return array0

    brick_collection_as_dict = {
        'visualizer': BrickUnbatched(callable=draw_function, input_names=['raw', 'arrays'], output_names=['visualized0'],
                                     permute_tensors=permute_tensors, tensors_to_numpy=tensors_to_numpy)
        }

    batch_size = 5
    model = BrickCollection(brick_collection_as_dict)
    list_of_arrays = [np.random.random((10, 20, 3)) for _ in range(batch_size)]
    outputs = model(named_inputs={'raw': torch.rand((batch_size, 3, 10, 20)), 'arrays': list_of_arrays}, stage=Stage.INFERENCE)

    assert len(outputs['visualized0']) == batch_size
    single_image = outputs['visualized0'][0]
    if tensors_to_numpy:
        assert isinstance(single_image, np.ndarray)
    else:
        assert isinstance(single_image, torch.Tensor)
    actual_shape = single_image.shape
    if permute_tensors:
        assert actual_shape == (10, 20, 3)
    else:
        assert actual_shape == (3, 10, 20)

def test_resolve_relative_names_errors():
    bricks = {
        'preprocessor': BrickModule(model=nn.Identity(), input_names=['raw'], output_names=['processed']),
        'backbone': BrickModule(model=nn.Identity(), input_names=['processed'], output_names=['embeddings']),
        'head0': {
            'classifier': BrickModule(model=nn.Identity(), input_names=['../../embeddings'], output_names=['./predictions']),
            'loss': BrickModule(model=nn.Identity(), input_names=['./predictions'], output_names=['./loss']),
        },
    }
    with pytest.raises(ValueError, match='Failed to resolve input name. Unable to resolve'):
        BrickCollection(bricks)

def test_no_inputs_or_outputs():

    class NoInputsNoOutputs(torch.nn.Module):
        def forward(self) -> None:
            return None

    bricks = {
        'preprocessor': BrickModule(model=NoInputsNoOutputs(), input_names=[], output_names=[]),
    }

    brick_collection = BrickCollection(bricks)
    brick_collection(named_inputs={'raw': torch.rand((2, 3, 100, 200))}, stage=Stage.INFERENCE)


def test_input_names_all():
    dict_bricks = create_brick_collection(num_classes=3, num_backbone_featues=5)

    class VisualizePredictions(torch.nn.Module):
        def forward(self, named_inputs: Dict[str, Any]):
            assert len(named_inputs) == 5
            return torch.concatenate((named_inputs['raw'], named_inputs['preprocessed']))

    dict_bricks['Visualize'] = bricks.BrickNotTrainable(VisualizePredictions(), input_names=['__all__'], output_names=['visualized'])
    brick_collection = BrickCollection(dict_bricks)
    brick_collection(named_inputs={'raw': torch.rand((2, 3, 100, 200))}, stage=Stage.INFERENCE)


def test_using_stage_inside_module():
    class StageDependentOutput(torch.nn.Module):
        def forward(self, name: str, stage: Stage) -> str:
            if stage == Stage.VALIDATION:
                return name + '_in_validation'
            return name + '_not_in_validation'

    brick_collection = BrickCollection(
        {
            'blah': bricks.BrickNotTrainable(StageDependentOutput(), input_names=['name', 'stage'], output_names=['output'])
        })
    named_outputs = brick_collection(named_inputs={'name': 'blah'}, stage=Stage.INFERENCE)
    assert named_outputs['output'] == 'blah_not_in_validation'

    named_outputs = brick_collection(named_inputs={'name': 'blah'}, stage=Stage.VALIDATION)
    assert named_outputs['output'] == 'blah_in_validation'


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
