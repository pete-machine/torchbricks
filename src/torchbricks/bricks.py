from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import torch
from torch import nn
from torchmetrics import MetricCollection, Metric

from torchbricks.bricks_helper import named_input_and_outputs_callable


class Phase(Enum):              # Gradients   Eval-model    Targets
    TRAIN = 'train'             # Y                   Y             Y
    VALIDATION = 'validation'   # N                   N             Y
    TEST = 'test'               # N                   N             Y
    INFERENCE = 'inference'     # N                   N             N
    EXPORT = 'export'           # N                   N             N



class BrickInterface(ABC):

    @abstractmethod
    def forward(self, named_inputs: Dict[str, Any], phase: Phase) -> Dict[str, Any]:
        """"""

    @abstractmethod
    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """"""

    @abstractmethod
    def summarize(self, phase: Phase, reset: bool) -> Dict[str, Any]:
        """"""


def parse_argument_loss_output_names(loss_output_names: Union[List[str], str], available_output_names: List[str]) -> List[str]:
    if loss_output_names == 'all':
        loss_output_names = available_output_names
    elif loss_output_names == 'none':
        loss_output_names = []

    assert isinstance(loss_output_names, List), f'`loss_output_names` should be `all`, `none` or a list of strings. {loss_output_names=}'
    assert set(loss_output_names).issubset(available_output_names), (f'One or more {loss_output_names=} is not an ',
                                                                      '`output_names` of brick {output_names=}')
    return loss_output_names


def parse_argument_run_on(run_on: Union[List[Phase], str]) -> List[Phase]:
    available_phases = list(Phase)
    if run_on == 'all':
        run_on = available_phases
    elif run_on == 'none':
        run_on = []

    assert isinstance(run_on, List), f'"run_on" should be "all", "none" or List[Phases]. It is {run_on=}'
    return run_on

class BrickModule(nn.Module, BrickInterface):
    def __init__(self, model: nn.Module,
                 input_names: Union[List[str], Dict[str, str], str],
                 output_names: List[str],
                 run_on: Union[List[Phase], str] = 'all',
                 loss_output_names: Union[List[str], str] = 'none',
                 calculate_gradients: bool = True,
                 trainable: Optional[bool] = None) -> None:
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.output_names = output_names

        self.loss_output_names = parse_argument_loss_output_names(loss_output_names, available_output_names=output_names)
        self.run_on = parse_argument_run_on(run_on)

        if calculate_gradients:
            self.calculate_gradients_on = [Phase.TRAIN]
        else:
            self.calculate_gradients_on = []

        trainable = trainable or True
        if not trainable and hasattr(model, 'requires_grad_'):
            self.model.requires_grad_(False)

    def forward(self, named_inputs: Dict[str, Any], phase: Phase) -> Dict[str, Any]:
        skip_forward = phase not in self.run_on
        if skip_forward:
            return {}

        calculate_gradients = phase in self.calculate_gradients_on
        named_inputs['phase'] = phase
        named_outputs = named_input_and_outputs_callable(callable=self.model, named_inputs=named_inputs, input_names=self.input_names,
                                                         output_names=self.output_names, calculate_gradients=calculate_gradients)

        return named_outputs

    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        named_losses = {name: loss for name, loss in named_outputs.items() if name in self.loss_output_names}
        return named_losses

    def summarize(self, phase: Phase, reset: bool) -> Dict[str, Any]:
        return {}



class BrickCollection(nn.ModuleDict, BrickInterface):  # Note BrickCollection is inherently ModuleDict and acts as a dictionary of modules
    def __init__(self, bricks: Dict[str, BrickInterface]) -> None:
        super().__init__(convert_nested_dict_to_nested_brick_collection(bricks))

    def forward(self, named_inputs: Dict[str, Any], phase: Phase, return_inputs: bool = True) -> Dict[str, Any]:
        gathered_named_io = dict(named_inputs)  # To keep the argument `named_inputs` unchanged
        for brick in self.values():
            results = brick.forward(phase=phase, named_inputs=gathered_named_io)
            gathered_named_io.update(results)

        if not return_inputs:
            [gathered_named_io.pop(name_input) for name_input in ['phase'] + list(named_inputs)]
        return gathered_named_io

    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        named_losses = {}
        for brick in self.values():
            named_losses.update(brick.extract_losses(named_outputs=named_outputs))
        return named_losses

    def summarize(self, phase: Phase, reset: bool) -> Dict[str, Any]:
        metrics = {}
        for brick in self.values():
            metrics.update(brick.summarize(phase=phase, reset=reset))
        return metrics


def convert_nested_dict_to_nested_brick_collection(bricks: Dict[str, Union[BrickInterface, Dict]], level=0):
    converted_bricks = {}
    for name, brick in bricks.items():
        if isinstance(brick, dict):
            converted_bricks[name] = convert_nested_dict_to_nested_brick_collection(brick, level=level+1)
        else:
            converted_bricks[name] = brick

    if level == 0:
        return converted_bricks
    else:
        return BrickCollection(converted_bricks)


class BrickTrainable(BrickModule):
    def __init__(self, model: nn.Module,
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str],
                 loss_output_names: Union[List[str], str] = 'none',
                 run_on: Optional[List[Phase]] = None):
        run_on = run_on or list(Phase)  # Runs on all phases
        super().__init__(model=model,
                         input_names=input_names,
                         output_names=output_names,
                         loss_output_names=loss_output_names,
                         run_on=run_on,
                         calculate_gradients=True,
                         trainable=True)


class BrickNotTrainable(BrickModule):
    def __init__(self, model: nn.Module,
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str],
                 run_on: Optional[List[Phase]] = None,
                 calculate_gradients: bool = True):
        run_on = run_on or list(Phase)  # Runs on all phases
        super().__init__(model=model,
                         input_names=input_names,
                         output_names=output_names,
                         loss_output_names='none',
                         run_on=run_on,
                         calculate_gradients=calculate_gradients,
                         trainable=False)


class BrickLoss(BrickModule):
    def __init__(self, model: nn.Module,
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str],
                 loss_output_names: Union[List[str], str] = 'all',
                 run_on: Optional[List[Phase]] = None,
                 ):

        run_on = run_on or [Phase.TRAIN, Phase.TEST, Phase.VALIDATION]
        super().__init__(model=model,
                         input_names=input_names,
                         output_names=output_names,
                         loss_output_names=loss_output_names,
                         run_on=run_on,
                         calculate_gradients=True,
                         trainable=True)


class BrickTorchMetric(BrickInterface, nn.Module):
    def __init__(self, metric: Union[MetricCollection, Metric],
                 input_names: Union[List[str], Dict[str, str]],
                 metric_name: Optional[str] = None,
                 run_on: Optional[List[Phase]] = None,
                 ):

        super().__init__()
        self.metric = metric
        self.input_names = input_names
        self.run_on = run_on or [Phase.TRAIN, Phase.TEST, Phase.VALIDATION]
        self.metric_name = metric_name or ''
        if self.metric_name == '' and isinstance(metric, Metric):
            raise ValueError(f'Specify `metric_name` when using {Metric}.')
        self.metrics_train = metric.clone()
        self.metrics_validation = metric.clone()
        self.metrics_test = metric.clone()

    def _select_metric_collection_from_split(self, phase: Phase) -> MetricCollection:
        if phase == Phase.TRAIN:
            return self.metrics_train
        elif phase == Phase.TEST:
            return self.metrics_test
        elif phase == Phase.VALIDATION:
            return self.metrics_validation
        raise TypeError('')

    @staticmethod
    def get_metric_name(phase: Phase, metric_name: str) -> str:
        return f'{phase.value}/{metric_name}'

    def forward(self, named_inputs: Union[List[str], Dict[str, str]], phase: Phase) -> Dict[str, Any]:
        skip_forward = phase not in self.run_on
        if skip_forward:
            return {}

        return named_input_and_outputs_callable(callable=self.metric.update, named_inputs=named_inputs, input_names=self.input_names,
                                                output_names=[], calculate_gradients=False)

    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def summarize(self, phase: Phase, reset: bool) -> Dict[str, Any]:
        metric = self._select_metric_collection_from_split(phase=phase)
        metrics = metric.compute()
        if reset:
            metric.reset()

        metric_name_prefix = self.get_metric_name(phase=phase, metric_name=self.metric_name)
        if isinstance(metric, MetricCollection):
            metrics = {f'{metric_name_prefix}{metric_name}': metric for metric_name, metric in metrics.items()}
        elif isinstance(metric, Metric):
            metrics = {metric_name_prefix: metrics}
        else:
            raise NameError()
        return metrics


class OnnxExportAdaptor(nn.Module):
    def __init__(self, model: nn.Module, phase: Phase) -> None:
        super().__init__()
        self.model = model
        self.phase = phase

    def forward(self, named_inputs):
        named_outputs = self.model.forward(named_inputs=named_inputs, phase=self.phase, return_inputs=False)
        return named_outputs


def export_as_onnx(brick_collection: BrickCollection,
                   named_inputs: Dict[str, torch.Tensor],
                   path_onnx: Path,
                   dynamic_batch_size: bool,
                   phase: Phase = Phase.EXPORT,
                   **onnx_export_kwargs):

    outputs = brick_collection(named_inputs=named_inputs, phase=phase, return_inputs=False)
    onnx_exportable = OnnxExportAdaptor(model=brick_collection, phase=phase)
    output_names = list(outputs)
    input_names = list(named_inputs)

    if dynamic_batch_size:
        if 'dynamic_axes' in onnx_export_kwargs:
            raise ValueError("Setting both 'dynamic_batch_size==True' and defining 'dynamic_axes' in 'onnx_export_kwargs' is not allowed. ")
        io_names = input_names + output_names
        dynamic_axes = {io_name: {0: 'batch_size'} for io_name in io_names}
        onnx_export_kwargs['dynamic_axes'] = dynamic_axes

    torch.onnx.export(model=onnx_exportable,
                      args=({'named_inputs': named_inputs}, ),
                      f=str(path_onnx),
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      **onnx_export_kwargs)
