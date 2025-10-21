from typing import Any, Callable, Dict, List, Union

import torch
from typeguard import typechecked

__ALL__ = "__all__"


def check_input_names(named_inputs: Dict[str, Any], input_names: List[str]):
    expected_names = [*list(named_inputs), __ALL__]
    is_subset = set(input_names).issubset(expected_names)
    assert is_subset, (
        f"Not all `{input_names=}` exists in `named_inputs={list(named_inputs)}`. The following expected names "
        f"{list(set(input_names).difference(named_inputs))} does not exist in the dictionary of `named_inputs`"
    )


def positional_arguments_from_list_input_names(
    named_inputs: Dict[str, Any], input_names: List[str]
) -> List:
    check_input_names(named_inputs=named_inputs, input_names=input_names)
    selected_inputs = [
        named_inputs if name == __ALL__ else named_inputs[name] for name in input_names
    ]
    return selected_inputs


def keyword_arguments_from_dict_input_names(named_inputs, input_names):
    input_name_keys = list(input_names.values())
    selected_inputs = positional_arguments_from_list_input_names(
        named_inputs, input_names=input_name_keys
    )
    argument_names_callable = list(input_names)
    arguments_and_values = dict(zip(argument_names_callable, selected_inputs))
    return arguments_and_values


def name_callable_outputs(outputs: Any, output_names: List[str]) -> Dict[str, Any]:
    if outputs is None:
        if len(output_names) == 0:
            return {}
        else:
            raise ValueError(
                f"No outputs was returned {outputs=} and we expected "
                f'"len(output_names)==0". However the following {output_names=} was specified'
            )

    if isinstance(outputs, dict):
        outputs = tuple(outputs.values())
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    assert len(outputs) == len(output_names), (
        f"The number of specified output names {output_names=} "
        f"does not match the actual number of outputs `{len(outputs)=}`"
    )
    return dict(zip(output_names, outputs))


@typechecked
def named_input_and_outputs_callable(
    callable: Callable,
    named_inputs: Dict[str, Any],
    input_names: Union[List[str], Dict[str, str]],
    output_names: List[str],
    calculate_gradients: bool = True,
) -> Dict[str, Any]:
    if isinstance(input_names, list):
        selected_inputs = positional_arguments_from_list_input_names(
            named_inputs, input_names=input_names
        )
        outputs = callable(*selected_inputs)
    elif isinstance(input_names, dict):
        arguments_and_values = keyword_arguments_from_dict_input_names(
            named_inputs, input_names
        )
        outputs = callable(**arguments_and_values)
    else:
        raise ValueError(
            f"`input_names` is not as expected `{input_names=}` should be a list of input names `List[str]`, a mapping from "
            "input names to function arguments `Dict[str, str]` or `all`"
        )
    if calculate_gradients:
        outputs = name_callable_outputs(outputs=outputs, output_names=output_names)
    else:
        with torch.no_grad():
            outputs = name_callable_outputs(outputs=outputs, output_names=output_names)
    return outputs


@typechecked
def parse_argument_loss_output_name_indices(
    loss_output_names: Union[List[str], str], available_output_names: List[str]
) -> List[int]:
    if loss_output_names == "all":
        loss_output_names = available_output_names
    elif loss_output_names == "none":
        loss_output_names = []

    assert isinstance(loss_output_names, List), (
        f"`loss_output_names` should be `all`, `none` or a list of strings. {loss_output_names=}"
    )
    assert set(loss_output_names).issubset(available_output_names), (
        f"One or more {loss_output_names=} is not an "
        "`output_names` of brick {output_names=}"
    )
    return [
        available_output_names.index(loss_output_name)
        for loss_output_name in loss_output_names
    ]
