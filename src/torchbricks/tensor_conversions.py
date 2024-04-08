import functools
from typing import List

import numpy as np
import torch
from PIL import Image
from typeguard import typechecked


def function_composer(*functions):
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

@typechecked
def unpack_batched_tensor_to_numpy_format(batched_tensor: torch.Tensor) -> List[np.ndarray]:
    """
    Converts a batched tensor with channel first to a list of numpy arrays with channel last

    Equals: function_composer(batched_tensor_to_channel_last, torch_to_numpy, unpack_batched_array_to_arrays)

    Examples:
    - [B, C, H, W] to list of [H, W, C] of length B
    - [B, C, H] to list of [H, C] of length B
    - [B, C] to list of [C] of length B
    - [B] to list of [1] of length B

    """
    np_array = batched_tensor_to_batched_np_array_channel_last(batched_tensor)
    return unpack_batched_array_to_arrays(np_array)



@typechecked
def unpack_batched_tensor_to_torchvision_format(batched_tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Converts a batched tensor to list of tensors

    Examples:
    - [B, C, H, W] to list of [C, H, W] of length B
    - [B, C, H] to list of [C, H] of length B
    - [B, C] to list of [C] of length B
    - [B] to list of [1] of length B
    """
    return list(batched_tensor)

@typechecked
def unpack_batched_tensor_to_pillow_images(batched_tensor: torch.Tensor) -> List[Image.Image]:
    """
    Converts a batched tensor to list of pillow images

    Examples:
    - [B, 3, H, W] to list of Images
    """
    assert batched_tensor.ndim == 4, f"Expected 4 dimensions, got {batched_tensor.ndim}"
    assert batched_tensor.shape[1] == 3, f"Expected 3 channels, got {batched_tensor.shape[1]}"
    np_array = batched_tensor_to_batched_np_array_channel_last(batched_tensor)
    return [Image.fromarray(float2uint8(np_array)) for np_array in np_array]


@typechecked
def unpack_batched_array_to_arrays(array: np.ndarray) -> List[np.ndarray]:
    """
    Converts a batched array to list of arrays

    Examples:
    - [B, H, W, C] to list of [H, W, C] of length B
    - [B, H, W] to list of [H, W] of length B
    - [B, H] to list of [H] of length B
    - [B] to list of [1] of length B

    """
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return list(array)

@typechecked
def batched_tensor_to_batched_np_array_channel_last(input_to_unpack: torch.Tensor) -> np.ndarray:
    input_to_unpack = batched_tensor_to_channel_last(input_to_unpack)
    return torch_to_numpy(input_to_unpack)


@typechecked
def batched_tensor_to_channel_last(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 3:
        return tensor

    permute_order_channel_last = [0, *list(range(2, tensor.ndim))] + [1]
    return tensor.permute(permute_order_channel_last)


@typechecked
def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

@typechecked
def float2uint8(array: np.ndarray) -> np.ndarray:
    return (array * 255).astype(np.uint8)
