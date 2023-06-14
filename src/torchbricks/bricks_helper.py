from typing import Any, Callable, Dict, List, Tuple, Union

def check_input_names(named_inputs: Dict[str, Any],
                      input_names: List[str]):
    is_subset = set(input_names).issubset(named_inputs)
    assert is_subset, (f"Not all '{input_names=}' exists in 'named_inputs={list(named_inputs)}'. The following expected names "
                       f"{list(set(input_names).difference(named_inputs))} does not exist in the dictionary of 'named_inputs'")


def select_inputs_by_name(named_inputs: Dict[str, Any],
                          input_names: List[str]) -> List:
    check_input_names(named_inputs=named_inputs, input_names=input_names)
    selected_inputs = [named_inputs[name] for name in input_names]
    return selected_inputs


def name_callable_outputs(outputs: Any,
                          output_names: List[str]) -> Dict[str, Any]:

    if not isinstance(outputs, Tuple):
        outputs = (outputs, )
    assert len(outputs) == len(output_names), (f'The number of specified output names {output_names=} '
                                               f"does not match the actual number of outputs '{len(outputs)=}'")
    return dict(zip(output_names, outputs))


def named_input_and_outputs_callable(callable: Callable,
                                     named_inputs: Dict[str, Any],
                                     input_names: Union[List[str], Dict[str, str]],
                                     output_names: List[str]) -> Dict[str, Any]:

    if isinstance(input_names, list):
        selected_inputs = select_inputs_by_name(named_inputs, input_names=input_names)
        outputs = callable(*selected_inputs)
    elif isinstance(input_names, dict):
        input_names_from_graph = list(input_names)
        input_names_to_callable = list(input_names.values())
        selected_inputs = select_inputs_by_name(named_inputs, input_names=input_names_from_graph)
        outputs = callable(**dict(zip(input_names_to_callable, selected_inputs)))
    else:
        raise ValueError(f"'input_names' is not as expected '{input_names=}' should be a list or dictionary.")

    return name_callable_outputs(outputs=outputs, output_names=output_names)
