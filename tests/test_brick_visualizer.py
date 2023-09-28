from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch
import torchbricks.brick_visualizer
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchbricks.brick_visualizer import BrickPerImageProcessing
from torchbricks.bricks import BrickCollection, BrickModule, Stage
from torchbricks.tensor_operations import unpack_batched_tensor_to_pillow_image
from typeguard import typechecked


def test_draw_image_classification():
    def draw_image_classification(input_image: Image.Image, target_name: str) -> Image.Image:
        """Draws image classification results"""
        draw = ImageDraw.Draw(input_image)
        font = ImageFont.truetype('tests/data/font_ASMAN.TTF', 50)
        draw.text((25, 25), text=target_name, font = font, align ='left')
        return input_image


    class BrickDrawImageClassification(BrickPerImageProcessing):
        def __init__(self, input_image: str, target_name: str, output_name: str):
            super().__init__(callable=draw_image_classification, input_names=[input_image, target_name], output_names=[output_name],
                             input_name_unpack_functions={input_image: unpack_batched_tensor_to_pillow_image})

    bricks = {
        'visualizer': BrickDrawImageClassification(input_image='input_image', target_name='target', output_name='visualization')
    }

    batched_inputs = {'input_image': torch.zeros((2, 3, 100, 200)), 'target': ['cat', 'dog']}
    brick_collection = BrickCollection(bricks)
    brick_collection(named_inputs=batched_inputs, stage=Stage.INFERENCE)


@pytest.mark.parametrize('input_names', [['raw', '__all__'], {'named_data': '__all__', 'array': 'raw'}])
def test_brick_per_image_processing_single_output_name(input_names):
    batch_size = 5
    unbatched_shape_raw = (3, 10, 20)
    image_raw = torch.rand((batch_size, *unbatched_shape_raw))
    named_inputs = {'raw': image_raw}
    expected_shape_raw = unbatched_shape_raw[1:] + unbatched_shape_raw[:1]

    @typechecked
    def draw_function(array: np.ndarray, named_data: Dict[str, Any]) -> np.ndarray:
        assert array.shape == expected_shape_raw
        assert set(named_data.keys()) == {'stage', 'raw', 'processed', 'embeddings'}
        return array

    brick_collection_as_dict = {
        'preprocessor': BrickModule(model=nn.Identity(), input_names=['raw'], output_names=['processed']),
        'backbone': BrickModule(model=nn.Identity(), input_names=['processed'], output_names=['embeddings']),
        'visualizer': BrickPerImageProcessing(callable=draw_function, input_names=input_names, output_names=['visualized'],
                                              type_unpack_functions=torchbricks.brick_visualizer.UNPACK_TENSORS_TO_NDARRAYS),
    }


    model = BrickCollection(brick_collection_as_dict)
    outputs = model(named_inputs=named_inputs, stage=Stage.INFERENCE)
    assert len(outputs['visualized']) == batch_size
    assert outputs['visualized'][0].shape == expected_shape_raw
    assert set(outputs) == {'raw', 'processed', 'embeddings', 'visualized', 'stage'}


@pytest.mark.parametrize('input_names', [['raw', 'processed'], {'tensor': 'processed', 'array0': 'raw'}])
def test_brick_per_image_processing_two_output_names_skip_unpack_functions_for(input_names):
    batch_size = 5
    unbatched_shape_raw = (3, 20, 20)
    image_raw = torch.rand((batch_size, *unbatched_shape_raw))
    expected_shape_raw = unbatched_shape_raw[1:] + unbatched_shape_raw[:1]

    unbatched_shape_processed = (3, 10, 20)
    image_processed = torch.rand((batch_size, *unbatched_shape_processed))
    expected_shape_processed = image_processed.shape # Expect original unbatched shape
    named_inputs = {'raw': image_raw,  'processed': image_processed}


    @typechecked
    def draw_function(array0: np.ndarray, tensor: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        assert array0.shape == expected_shape_raw
        assert tensor.shape == expected_shape_processed
        return array0, tensor


    brick_collection_as_dict = {
        'backbone': BrickModule(model=nn.Identity(), input_names=['processed'], output_names=['embeddings']),
        'visualizer': BrickPerImageProcessing(callable=draw_function, input_names=input_names,
                                     output_names=['visualized0', 'visualized1'],
                                     type_unpack_functions=torchbricks.brick_visualizer.UNPACK_TENSORS_TO_NDARRAYS,
                                     input_name_unpack_functions={'processed': None}),
    }


    model = BrickCollection(brick_collection_as_dict)
    outputs = model(named_inputs=named_inputs, stage=Stage.INFERENCE)
    assert len(outputs['visualized0']) == batch_size
    assert len(outputs['visualized1']) == batch_size
    assert outputs['visualized0'][0].shape == expected_shape_raw
    assert outputs['visualized1'][0].shape == expected_shape_processed


@pytest.mark.parametrize('input_names', [['raw', 'processed'], {'tensor1': 'processed', 'tensor0': 'raw'}])
def test_brick_per_image_processing_two_output_names_no_torch_to_numpy_unpacking(input_names):
    batch_size = 5
    unbatched_shape_raw = (3, 20, 20)
    image_raw = torch.rand((batch_size, *unbatched_shape_raw))
    expected_shape_raw = unbatched_shape_raw

    unbatched_shape_processed = (3, 10, 20)
    image_processed = torch.rand((batch_size, *unbatched_shape_processed))
    expected_shape_processed = image_processed.shape # Expect original unbatched shape
    named_inputs = {'raw': image_raw,  'processed': image_processed}

    @typechecked
    def draw_function(tensor0: torch.Tensor, tensor1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert tensor0.shape == expected_shape_raw
        assert tensor1.shape == expected_shape_processed
        return tensor0, tensor1


    brick_collection_as_dict = {
        'backbone': BrickModule(model=nn.Identity(), input_names=['processed'], output_names=['embeddings']),
        'visualizer': BrickPerImageProcessing(callable=draw_function, input_names=input_names,
                                     output_names=['visualized0', 'visualized1'],
                                     type_unpack_functions=torchbricks.brick_visualizer.UNPACK_NO_CONVERSION,
                                     input_name_unpack_functions={'processed': None}),
    }


    model = BrickCollection(brick_collection_as_dict)
    outputs = model(named_inputs=named_inputs, stage=Stage.INFERENCE)
    assert len(outputs['visualized0']) == batch_size
    assert len(outputs['visualized1']) == batch_size
    assert outputs['visualized0'][0].shape == expected_shape_raw
    assert outputs['visualized1'][0].shape == expected_shape_processed
