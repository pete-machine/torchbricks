from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Set, Union

import numpy as np
import torch
from typeguard import typechecked

from torchbricks.brick_tags import Tag
from torchbricks.bricks import BrickModule, use_default_style
from torchbricks.bricks_helper import name_callable_outputs
from torchbricks.tensor_conversions import unpack_batched_array_to_arrays, unpack_batched_tensor_to_numpy_format

UNPACK_NO_CONVERSION = {
    torch.Tensor: list,  # Unpack as list only [B, C, H, W] -> [C, H, W]
    np.ndarray: unpack_batched_array_to_arrays,  # Unpack as list only [B, H, W, C] -> [H, W, C]
}

UNPACK_TENSORS_TO_NDARRAYS = {
    torch.Tensor: unpack_batched_tensor_to_numpy_format,
    np.ndarray: unpack_batched_array_to_arrays,  # Unpack as list only
}


@typechecked
class BrickPerImageVisualization(BrickModule):
    """
    Primarily used to visualize data per image.
    The callable-function is called for each image in the batch and the outputs are collected in a list.

    Brick to perform operations on a batched input to a list of unbatched inputs.
    Useful when during per image visualization and processing.
    """

    style: Dict[str, str] = use_default_style({"fill": "#5C677D"})

    def __init__(
        self,
        callable: Callable,
        input_names: Union[List[str], Dict[str, str]],
        output_names: List[str],
        unpack_functions_for_type: Optional[Dict[type, Optional[Callable]]] = None,
        tags: Union[Set[str], List[str], str] = Tag.VISUALIZATION,
        unpack_functions_for_input_name: Optional[Dict[str, Optional[Callable]]] = None,
    ):
        """
        Parameters
        ----------
        type_unpack_functions : Dict[type, Callable]
            Specifies type specific functions to unpack data.
            Example: To convert a torch.tensor of shape [B, C, H, W] to numpy.array of shape [B, H, W, C]
                type_unpack_functions = {
                    torch.Tensor: unpack_batched_tensor_to_numpy_array_channel_last
                    }
            Two common use cases:
            1) `type_unpack_functions=UNPACK_TENSORS_TO_NDARRAYS`
                UNPACK_TENSORS_TO_NDARRAYS = {
                    torch.Tensor: unpack_batched_tensor_to_numpy_channel_last_arrays,
                    np.ndarray: list,  # Unpack as list only
                }
                to unpack tensors channel first to numpy arrays channel last.
               Converts a torch.tensor of shape [B, C, H, W] to numpy.array of shape [B, H, W, C]. This is then iterated over
               and the callable-function receives numpy array of shape [H, W, C].
            2) `type_unpack_functions=UNPACK_NO_CONVERSION` to keep torch tensors as they are.
               E.g. A torch tensor of shape [B, C, H, W] will keep type and shape. This is then iterated over
               and the callable-function receives a torch tensor of shape [C, H, W].

        """
        super().__init__(
            model=self.unpack_data,
            input_names=input_names,
            output_names=output_names,
            loss_output_names=[],
            tags=tags,
            calculate_gradients=False,
            trainable=False,
        )

        self.type_unpack_functions = unpack_functions_for_type or UNPACK_TENSORS_TO_NDARRAYS
        self.callable = callable

        self.input_name_unpack_functions = unpack_functions_for_input_name or {}
        input_names_list = self.input_names_as_list()
        input_names_to_unpack = list(self.input_name_unpack_functions)
        assert set(input_names_to_unpack).issubset(input_names_list), (
            f"One or more {input_names_to_unpack=} is not an ",
            f"`input_names` of brick {input_names_list=}",
        )

    def unpack_data(self, *args, **kwargs):
        uses_keyword_args = len(kwargs) > 0

        if uses_keyword_args:
            function_kwargs = kwargs.values()
        else:
            function_kwargs = args
        named_data_batched = dict(zip(self.input_names_as_list(), function_kwargs))
        named_data_unpacked_as_lists = {}
        for input_name, input_to_unpack in named_data_batched.items():
            unpack_function = self.type_unpack_functions.get(type(input_to_unpack), None)
            if input_name in self.input_name_unpack_functions:  # input_name unpack functions are prioritized above type unpack functions
                unpack_function = self.input_name_unpack_functions[input_name]

            if unpack_function is not None:
                input_to_unpack = unpack_function(input_to_unpack)
            named_data_unpacked_as_lists[input_name] = input_to_unpack

        unpack_types = (list, tuple)
        batch_sizes = [len(unpacked) for unpacked in named_data_unpacked_as_lists.values() if isinstance(unpacked, unpack_types)]
        if len(batch_sizes) == 0:
            raise ValueError(f"Can not estimate batch size from these inputs: {self.input_names_as_list()}")

        batch_size_counts = Counter(batch_sizes)
        if len(batch_size_counts) > 1:
            raise ValueError(f"Batch size is not the same for all inputs: {self.input_names_as_list()}")

        batch_size = batch_size_counts.most_common(1)[0][0]
        per_image_data = defaultdict(dict)
        for input_name, input_unpacked_as_list in named_data_unpacked_as_lists.items():
            safe_to_unpack = isinstance(input_unpacked_as_list, unpack_types) and (len(input_unpacked_as_list) == batch_size)
            if safe_to_unpack:
                for i_image, function_kwargs in enumerate(input_unpacked_as_list):
                    per_image_data[i_image][input_name] = function_kwargs
            else:
                # If we can not unpack data, we pass it as it is (e.g. a string or dictionary or data with incorrect length/batch size).
                for i_image in range(batch_size):
                    per_image_data[i_image][input_name] = input_unpacked_as_list

        outputs_per_image = defaultdict(list)
        for i_image in range(batch_size):
            if uses_keyword_args:
                function_kwargs = {arg_name: per_image_data[i_image][input_name] for arg_name, input_name in self.input_names.items()}
                image_outputs = self.callable(**function_kwargs)
            else:
                image_outputs = self.callable(*per_image_data[i_image].values())

            named_outputs = name_callable_outputs(outputs=image_outputs, output_names=self.output_names)
            for output_name, output in named_outputs.items():
                outputs_per_image[output_name].append(output)

        return tuple(outputs_per_image.values())
