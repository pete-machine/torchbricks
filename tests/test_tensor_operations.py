import pytest
import torch
from torchbricks.tensor_operations import permute_batched_tensor_to_channel_last, unpack_batched_tensor_to_numpy_channel_last_arrays


@pytest.mark.parametrize(['tensor_shape', 'tensor_shape_expected'],
                         [((5,), (5,)),  # Same shape
                          ((5, 3), (5, 3)),  # Same shape
                          ((5, 3, 10), (5, 10, 3)),
                          ((5, 3, 10, 20), (5, 10, 20, 3)),
                          ((5, 3, 10, 20, 30), (5, 10, 20, 30, 3))])
def test_permute_batched_tensor_to_channel_last(tensor_shape, tensor_shape_expected):
    tensor = torch.ones(tensor_shape)
    result = permute_batched_tensor_to_channel_last(tensor)
    assert result.shape == tensor_shape_expected

@pytest.mark.parametrize(['tensor_shape', 'tensor_shape_expected'],
                         [((5,), (5,)),  # Same shape
                          ((5, 3), (5, 3)),  # Same shape
                          ((5, 3, 10), (5, 10, 3)),
                          ((5, 3, 10, 20), (5, 10, 20, 3)),
                          ((5, 3, 10, 20, 30), (5, 10, 20, 30, 3))])
def test_unpack_batched_tensor_to_numpy_channel_last_arrays(tensor_shape, tensor_shape_expected):
    tensor = torch.ones(tensor_shape)
    result = unpack_batched_tensor_to_numpy_channel_last_arrays(tensor)
    assert len(result) == tensor_shape_expected[0]
    assert result[0].shape == tensor_shape_expected[1:]
