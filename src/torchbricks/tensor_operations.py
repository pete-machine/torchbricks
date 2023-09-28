from typing import List

import numpy as np
import torch
from PIL import Image
from typeguard import typechecked


@typechecked
def unpack_batched_tensor_to_numpy_channel_last_arrays(input_to_unpack: torch.Tensor) -> List[np.ndarray]:
    np_array = _unpack_batched_tensor_to_numpy_channel_last_array(input_to_unpack)
    if np_array.ndim == 1:  # A 1D array of shape N, returns a list with N elements of shape 1 numpy arrays
        return [np.array(value) for value in np_array]
    return list(np_array)

@typechecked
def unpack_batched_tensor_to_pillow_image(input_to_unpack: torch.Tensor) -> List[Image.Image]:
    np_array = _unpack_batched_tensor_to_numpy_channel_last_array(input_to_unpack)
    return [Image.fromarray(float2uint8(np_array)) for np_array in np_array]

@typechecked
def _unpack_batched_tensor_to_numpy_channel_last_array(input_to_unpack: torch.Tensor) -> np.ndarray:
    input_to_unpack = permute_batched_tensor_to_channel_last(input_to_unpack)
    return tensor_as_numpy_array(input_to_unpack)


@typechecked
def permute_batched_tensor_to_channel_last(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 3:
        return tensor

    permute_order_channel_last = [0, *list(range(2, tensor.ndim))] + [1]
    return tensor.permute(permute_order_channel_last)


@typechecked
def tensor_as_numpy_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

def float2uint8(array: np.ndarray) -> np.ndarray:
    return (array * 255).astype(np.uint8)
