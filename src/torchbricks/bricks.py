from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from torchmetrics import Metric, MetricCollection

from torchbricks.bricks_helper import named_input_and_outputs_callable


class Stage(Enum):              # Gradients   Eval-model    Targets
    TRAIN = 'train'             # Y                   Y             Y
    VALIDATION = 'validation'   # N                   N             Y
    TEST = 'test'               # N                   N             Y
    INFERENCE = 'inference'     # N                   N             N
    EXPORT = 'export'           # N                   N             N


class BrickInterface(ABC):
    def __init__(self,
                 input_names: Union[List[str], Dict[str, str], str],
                 output_names: List[str],
                 alive_stages: Union[List[Stage], str] = 'all',
                 ) -> None:
        super().__init__()
        self.input_names = input_names
        self.output_names = output_names
        self.alive_stages: List[Stage] = parse_argument_alive_stages(alive_stages)

    def run_now(self, stage: Stage) -> bool:
        return stage in self.alive_stages

    @abstractmethod
    def forward(self, named_inputs: Dict[str, Any], stage: Stage) -> Dict[str, Any]:
        """"""

    @abstractmethod
    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """"""

    @abstractmethod
    def summarize(self, stage: Stage, reset: bool) -> Dict[str, Any]:
        """"""


def parse_argument_alive_stages(alive_stages: Union[List[Stage], str]) -> List[Stage]:
    available_stages = list(Stage)
    if alive_stages == 'all':
        alive_stages = available_stages
    elif alive_stages == 'none':
        alive_stages = []

    assert isinstance(alive_stages, List), f'"alive_stages" should be "all", "none" or List[stages]. It is {alive_stages=}'
    return alive_stages


def parse_argument_loss_output_names(loss_output_names: Union[List[str], str], available_output_names: List[str]) -> List[str]:
    if loss_output_names == 'all':
        loss_output_names = available_output_names
    elif loss_output_names == 'none':
        loss_output_names = []

    assert isinstance(loss_output_names, List), f'`loss_output_names` should be `all`, `none` or a list of strings. {loss_output_names=}'
    assert set(loss_output_names).issubset(available_output_names), (f'One or more {loss_output_names=} is not an ',
                                                                      '`output_names` of brick {output_names=}')
    return loss_output_names


class BrickModule(nn.Module, BrickInterface):
    def __init__(self, model: nn.Module,
                 input_names: Union[List[str], Dict[str, str], str],
                 output_names: List[str],
                 alive_stages: Union[List[Stage], str] = 'all',
                 loss_output_names: Union[List[str], str] = 'none',
                 calculate_gradients: bool = True,
                 trainable: bool = True) -> None:
        nn.Module.__init__(self)
        BrickInterface.__init__(self, input_names=input_names, output_names=output_names, alive_stages=alive_stages)
        self.model = model
        self.loss_output_names = parse_argument_loss_output_names(loss_output_names, available_output_names=output_names)

        if calculate_gradients:
            self.calculate_gradients_on = [Stage.TRAIN]
        else:
            self.calculate_gradients_on = []

        if not trainable and hasattr(model, 'requires_grad_'):
            self.model.requires_grad_(False)

    def calculate_gradients(self, stage: Stage) -> bool:
        return stage in self.calculate_gradients_on

    def forward(self, named_inputs: Dict[str, Any], stage: Stage) -> Dict[str, Any]:
        if not self.run_now(stage=stage):
            return {}


        named_inputs['stage'] = stage
        named_outputs = named_input_and_outputs_callable(callable=self.model,
                                                         named_inputs=named_inputs,
                                                         input_names=self.input_names,
                                                         output_names=self.output_names,
                                                         calculate_gradients=self.calculate_gradients(stage=stage))

        return named_outputs

    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        named_losses = {name: loss for name, loss in named_outputs.items() if name in self.loss_output_names}
        return named_losses

    def summarize(self, stage: Stage, reset: bool) -> Dict[str, Any]:
        return {}



class BrickCollection(nn.ModuleDict):  # Note BrickCollection is inherently ModuleDict and acts as a dictionary of modules
    def __init__(self, bricks: Dict[str, BrickInterface]) -> None:
        super().__init__(convert_nested_dict_to_nested_brick_collection(bricks))

    def forward(self, named_inputs: Dict[str, Any], stage: Stage, return_inputs: bool = True) -> Dict[str, Any]:
        gathered_named_io = dict(named_inputs)  # To keep the argument `named_inputs` unchanged

        for brick in self.values():
            if brick.run_now(stage=stage):
                results = brick.forward(stage=stage, named_inputs=gathered_named_io)
                gathered_named_io.update(results)

        if not return_inputs:
            [gathered_named_io.pop(name_input) for name_input in ['stage', *list(named_inputs)]]
        return gathered_named_io

    def run_now(self, stage: Stage) -> bool:
        return True

    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        named_losses = {}
        for brick in self.values():
            named_losses.update(brick.extract_losses(named_outputs=named_outputs))
        return named_losses

    def summarize(self, stage: Stage, reset: bool) -> Dict[str, Any]:
        metrics = {}
        for brick in self.values():
            metrics.update(brick.summarize(stage=stage, reset=reset))
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
                 alive_stages: Optional[List[Stage]] = None):
        alive_stages = alive_stages or list(Stage)  # Runs on all stages
        super().__init__(model=model,
                         input_names=input_names,
                         output_names=output_names,
                         loss_output_names=loss_output_names,
                         alive_stages=alive_stages,
                         calculate_gradients=True,
                         trainable=True)


class BrickNotTrainable(BrickModule):
    def __init__(self, model: nn.Module,
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str],
                 alive_stages: Optional[List[Stage]] = None,
                 calculate_gradients: bool = True):
        alive_stages = alive_stages or list(Stage)  # Runs on all stages
        super().__init__(model=model,
                         input_names=input_names,
                         output_names=output_names,
                         loss_output_names='none',
                         alive_stages=alive_stages,
                         calculate_gradients=calculate_gradients,
                         trainable=False)


class BrickLoss(BrickModule):
    def __init__(self, model: nn.Module,
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str],
                 loss_output_names: Union[List[str], str] = 'all',
                 alive_stages: Optional[List[Stage]] = None,
                 ):

        alive_stages = alive_stages or [Stage.TRAIN, Stage.TEST, Stage.VALIDATION]
        super().__init__(model=model,
                         input_names=input_names,
                         output_names=output_names,
                         loss_output_names=loss_output_names,
                         alive_stages=alive_stages,
                         calculate_gradients=True,
                         trainable=True)


class BrickMetricMultiple(BrickModule):
    def __init__(self, metric_collection: MetricCollection,
                 input_names: Union[List[str], Dict[str, str]],
                 alive_stages: Optional[List[Stage]] = None,
                 return_metrics: bool = False,
                 ):
        alive_stages = alive_stages or [Stage.TRAIN, Stage.TEST, Stage.VALIDATION]
        if return_metrics:
            output_names = list(metric_collection)
        else:
            output_names = []
        self.return_metrics = return_metrics
        metrics = nn.ModuleDict({stage.name: metric_collection.clone() for stage in alive_stages})
        super().__init__(model=metrics, input_names=input_names, output_names=output_names, alive_stages=alive_stages,
                         trainable=False,
                         calculate_gradients=False)

    def forward(self, named_inputs: Union[List[str], Dict[str, str]], stage: Stage) -> Dict[str, Any]:
        skip_forward = stage not in self.alive_stages
        if skip_forward:
            return {}

        metric_collection = self.model[stage.name]
        if self.return_metrics:
            output_names = ['metrics']
            metric_callable = metric_collection  # Return metrics as a dictionary
        else:
            output_names = []
            metric_callable = metric_collection.update  # Will not return metrics

        output = named_input_and_outputs_callable(callable=metric_callable,
                                                  named_inputs=named_inputs,
                                                  input_names=self.input_names,
                                                  output_names=output_names,
                                                  calculate_gradients=self.calculate_gradients(stage=stage))
        if self.return_metrics:
            return output['metrics']  # Metrics in a dictionary
        else:
            assert output == {}
            return {}

    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def summarize(self, stage: Stage, reset: bool) -> Dict[str, Any]:
        metric_collection = self.model[stage.name]
        metrics = metric_collection.compute()
        if reset:
            metric_collection.reset()

        return metrics

class BrickMetricSingle(BrickMetricMultiple):
    def __init__(self,
                 metric: Metric,
                 input_names: List[str] | Dict[str, str],
                 metric_name: Optional[str],
                 alive_stages: List[Stage] | None = None,
                 return_metrics: bool = False):
        metric_name = metric_name or metric.__class__.__name__
        metric_collection = MetricCollection({metric_name: metric})
        super().__init__(metric_collection=metric_collection, input_names=input_names, alive_stages=alive_stages,
                         return_metrics=return_metrics)


class OnnxExportAdaptor(nn.Module):
    def __init__(self, model: nn.Module, stage: Stage) -> None:
        super().__init__()
        self.model = model
        self.stage = stage

    def forward(self, named_inputs):
        named_outputs = self.model.forward(named_inputs=named_inputs, stage=self.stage, return_inputs=False)
        return named_outputs


def export_as_onnx(brick_collection: BrickCollection,
                   named_inputs: Dict[str, torch.Tensor],
                   path_onnx: Path,
                   dynamic_batch_size: bool,
                   stage: Stage = Stage.EXPORT,
                   **onnx_export_kwargs):

    outputs = brick_collection(named_inputs=named_inputs, stage=stage, return_inputs=False)
    onnx_exportable = OnnxExportAdaptor(model=brick_collection, stage=stage)
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
