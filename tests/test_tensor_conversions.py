import numpy as np
import pytest
import torch
from PIL import Image
from torchbricks.tensor_conversions import (
    batched_tensor_to_channel_last,
    function_composer,
    unpack_batched_tensor_to_numpy_format,
    unpack_batched_tensor_to_pillow_images,
    unpack_batched_tensor_to_torchvision_format,
)


@pytest.mark.parametrize(['tensor_shape', 'tensor_shape_expected'],
                         [((5,), (5,)),  # Same shape
                          ((5, 3), (5, 3)),  # Same shape
                          ((5, 3, 10), (5, 10, 3)),
                          ((5, 3, 10, 20), (5, 10, 20, 3)),
                          ((5, 3, 10, 20, 30), (5, 10, 20, 30, 3))])
def test_batched_tensor_to_channel_last(tensor_shape, tensor_shape_expected):
    tensor = torch.ones(tensor_shape)
    result = batched_tensor_to_channel_last(tensor)
    assert result.shape == tensor_shape_expected

@pytest.mark.parametrize(['tensor_shape', 'tensor_full_shape_expected'],
                         [((5,), (5,)),  # Same shape
                          ((5, 3), (5, 3)),  # Same shape
                          ((5, 3, 10), (5, 10, 3)),
                          ((5, 3, 10, 20), (5, 10, 20, 3)),
                          ((5, 3, 10, 20, 30), (5, 10, 20, 30, 3))])
def test_unpack_batched_tensor_to_numpy_format(tensor_shape, tensor_full_shape_expected):
    tensor = torch.ones(tensor_shape)
    result = unpack_batched_tensor_to_numpy_format(tensor)
    assert len(result) == tensor_full_shape_expected[0]
    if len(tensor_full_shape_expected) == 1:
        tensor_shape_expected = (1, )
    else:
        tensor_shape_expected = tensor_full_shape_expected[1:]
    assert result[0].shape == tensor_shape_expected
    assert isinstance(result[0], np.ndarray)

@pytest.mark.parametrize(['tensor_shape', 'tensor_full_shape_expected'],
                         [((5,), (5,)),  # Same shape
                          ((5, 3), (5, 3)),  # Same shape
                          ((5, 3, 10), (5, 3, 10)),
                          ((5, 3, 10, 20), (5, 3, 10, 20)),
                          ((5, 3, 10, 20, 30), (5, 3, 10, 20, 30))])
def test_unpack_batched_tensor_to_torchvision_format(tensor_shape, tensor_full_shape_expected):
    tensor = torch.ones(tensor_shape)
    result = unpack_batched_tensor_to_torchvision_format(tensor)
    assert len(result) == tensor_full_shape_expected[0]
    tensor_shape_expected = tensor_full_shape_expected[1:]
    assert result[0].shape == tensor_shape_expected
    assert isinstance(result[0], torch.Tensor)



def test_unpack_batched_tensor_to_pillow_images():
    tensor_shape = (5, 3, 10, 20)
    tensor_full_shape_expected = (5, 10, 20, 3)
    tensor = torch.ones(tensor_shape)
    result = unpack_batched_tensor_to_pillow_images(tensor)
    assert len(result) == tensor_full_shape_expected[0]
    assert isinstance(result[0], Image.Image)
    assert (result[0].height, result[0].width) == tensor_full_shape_expected[1:3]

@pytest.mark.parametrize('tensor_shape', [(5, 3, 10), (5, 4, 10, 20)])
def test_unpack_batched_tensor_to_pillow_images_fail(tensor_shape):
    tensor = torch.ones(tensor_shape)

    with pytest.raises(AssertionError):
        unpack_batched_tensor_to_pillow_images(tensor)

def test_function_composer():
    tensor = torch.ones(3)

    def multiplier_by_two(x):
        return x * 2

    composed_function = function_composer(multiplier_by_two, list)
    result = composed_function(tensor)
    assert result == [2, 2, 2]
