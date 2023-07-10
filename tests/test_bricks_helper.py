
from typing import Tuple

import pytest
from torchbricks.bricks_helper import named_input_and_outputs_callable


def test_named_input_and_outputs_callable_single_in_single_out():
    def x2(value: float) -> float:
        return value*2

    named_inputs = {'in': 3}
    named_outputs = named_input_and_outputs_callable(x2, named_inputs=named_inputs, input_names=['in'], output_names=['out'])
    assert named_outputs == {'out': 6}

    with pytest.raises(AssertionError, match='Not all `input_names='):
        named_input_and_outputs_callable(x2, named_inputs=named_inputs,  input_names=['not_a_name'], output_names=['out'])

    with pytest.raises(AssertionError, match='Not all `input_names='):
        named_input_and_outputs_callable(x2, named_inputs=named_inputs,  input_names=['in', 'in1'], output_names=['out'])

    named_outputs = named_input_and_outputs_callable(x2, named_inputs=named_inputs, input_names={'in': 'value'}, output_names=['out'])
    assert named_outputs == {'out': 6}

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'not_a_name'"):
        named_input_and_outputs_callable(x2, named_inputs=named_inputs, input_names={'in': 'not_a_name'}, output_names=['out'])

    # Too few output arguments
    with pytest.raises(AssertionError, match='The number of specified output names'):
        named_input_and_outputs_callable(x2, named_inputs=named_inputs, input_names=['in'], output_names=[])

    # Too many output arguments
    with pytest.raises(AssertionError, match='The number of specified output names'):
        named_input_and_outputs_callable(x2, named_inputs=named_inputs, input_names=['in'], output_names=['out0', 'out1'])

def test_named_input_and_outputs_callable_two_in_two_out():
    def add_subtract(value0: float, value1: float) -> Tuple[float, float]:
        return value0 + value1, value0-value1

    named_inputs = {'in0': 3, 'in1': 1}
    named_outputs = named_input_and_outputs_callable(add_subtract,
                                                     named_inputs=named_inputs,
                                                     input_names=['in0', 'in1'],
                                                     output_names=['out0', 'out1'])
    assert named_outputs == {'out0': 4, 'out1': 2}

    named_outputs = named_input_and_outputs_callable(add_subtract,
                                                     named_inputs=named_inputs,
                                                     input_names={'in0': 'value0', 'in1': 'value1'},
                                                     output_names=['out0','out1'])
    assert named_outputs == {'out0': 4, 'out1': 2}

    named_outputs = named_input_and_outputs_callable(add_subtract,
                                                     named_inputs=named_inputs,
                                                     input_names={'in1': 'value1', 'in0': 'value0'},
                                                     output_names=['out0','out1'])
    assert named_outputs == {'out0': 4, 'out1': 2}

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'not_a_name'"):
            named_outputs = named_input_and_outputs_callable(add_subtract,
                                                     named_inputs=named_inputs,
                                                     input_names={'in0': 'not_a_name', 'in1': 'value1'},
                                                     output_names=['out0','out1'])


    with pytest.raises(AssertionError, match='The number of specified output names'):
            named_outputs = named_input_and_outputs_callable(add_subtract,
                                                     named_inputs=named_inputs,
                                                     input_names={'in0': 'value0', 'in1': 'value1'},
                                                     output_names=['out0'])
