import inspect
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from torch import nn
from torchmetrics import Metric, MetricCollection
from typeguard import typechecked

from torchbricks.bricks_helper import named_input_and_outputs_callable


class Stage(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'
    INFERENCE = 'inference'
    EXPORT = 'export'

def use_default_style(overwrites: Optional[Dict[str, str]] = None):
    overwrites = overwrites or {}
    default_style = {'stroke-width': '0px'}
    default_style.update(overwrites)
    return default_style

class BrickInterface(ABC):
    # 67635c,f5e26b,f1dfca,db9d38,dc9097,c4779f,c75239,84bb84,394a89,7d9dc4
    style: Dict[str, str] = use_default_style()
    def __init__(self,
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str],
                 alive_stages: Union[List[Stage], str]) -> None:
        super().__init__()

        self.input_names = input_names
        self.output_names = output_names
        self.alive_stages: List[Stage] = parse_argument_alive_stages(alive_stages)

    def run_now(self, stage: Stage) -> bool:
        return stage in self.alive_stages

    def __call__(self, named_inputs: Dict[str, Any], stage: Stage) -> Dict[str, Any]:
        return self.forward(named_inputs=named_inputs, stage=stage)

    @abstractmethod
    def forward(self, named_inputs: Dict[str, Any], stage: Stage) -> Dict[str, Any]:
        """"""

    @abstractmethod
    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """"""

    @abstractmethod
    def get_module_name(self) -> str:
        """"""

    def summarize(self, stage: Stage, reset: bool) -> Dict[str, Any]:
        if hasattr(self.model, 'summarize') and inspect.isfunction(self.model.summarize):
            return self.model(stage=stage, reset=reset)
        return {}

    def get_brick_type(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        input_names = self.input_names
        output_names = self.output_names
        alive_stages = [phase.name for phase in self.alive_stages]
        return f'{self.get_brick_type()}({self.get_module_name()}, {input_names=}, {output_names=}, {alive_stages=})'



@typechecked
def parse_argument_alive_stages(alive_stages: Union[List[Stage], str]) -> List[Stage]:
    available_stages = list(Stage)
    if alive_stages == 'all':
        alive_stages = available_stages
    elif alive_stages == 'none':
        alive_stages = []

    assert isinstance(alive_stages, List), f'"alive_stages" should be "all", "none" or List[stages]. It is {alive_stages=}'
    return alive_stages


@typechecked
def parse_argument_loss_output_names(loss_output_names: Union[List[str], str], available_output_names: List[str]) -> List[str]:
    if loss_output_names == 'all':
        loss_output_names = available_output_names
    elif loss_output_names == 'none':
        loss_output_names = []

    assert isinstance(loss_output_names, List), f'`loss_output_names` should be `all`, `none` or a list of strings. {loss_output_names=}'
    assert set(loss_output_names).issubset(available_output_names), (f'One or more {loss_output_names=} is not an ',
                                                                      '`output_names` of brick {output_names=}')
    return loss_output_names


@typechecked
class BrickModule(nn.Module, BrickInterface):
    style: Dict[str, str] = use_default_style({'fill' :'#355070'})
    def __init__(self, model: Union[nn.Module, nn.ModuleDict, Callable],
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str],
                 alive_stages: Union[List[Stage], str] = 'all',
                 loss_output_names: Union[List[str], str] = 'none',
                 calculate_gradients: bool = True,
                 trainable: bool = True) -> None:
        nn.Module.__init__(self)
        BrickInterface.__init__(self,
                                input_names=input_names,
                                output_names=output_names,
                                alive_stages=alive_stages)
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

    def get_module_name(self) -> str:
        return self.model.__class__.__name__

    def __repr__(self):  # Overwrite '__repr__' of 'BrickModule'
        return BrickInterface.__repr__(self)


@typechecked
class BrickCollection(nn.ModuleDict):
    def __init__(self, bricks: Dict[str, BrickInterface]) -> None:
        resolve_relative_names(bricks=bricks)
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

@typechecked
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


@typechecked
def resolve_relative_names(bricks: Dict[str, BrickInterface]):
    return _resolve_relative_names_recursive(bricks)

@typechecked
def _resolve_relative_names_recursive(bricks: Union[Dict[str, BrickInterface], BrickCollection], parent: Optional[Path] = None):
    parent = parent or Path('/root')
    for brick_name, brick_or_bricks in bricks.items():
        if isinstance(brick_or_bricks, (dict, BrickCollection)):
            _resolve_relative_names_recursive(bricks=brick_or_bricks, parent=parent / brick_name)
        elif isinstance(brick_or_bricks, BrickInterface):
            brick_or_bricks.input_names = _resolve_input_or_output_names(brick_or_bricks.input_names, parent)
            brick_or_bricks.output_names = _resolve_input_or_output_names(brick_or_bricks.output_names, parent)
        else:
            raise ValueError('')

    return bricks

@typechecked
def _resolve_input_or_output_names(input_or_output_names: Union[List[str], Dict[str, str]],
                                   parent: Path) -> Union[List[str], Dict[str, str]]:

    if isinstance(input_or_output_names, dict):
        data_names = input_or_output_names.values()
    else:
        data_names = list(input_or_output_names)
    input_names_resolved = []
    for input_name in data_names:
        is_relative_name = input_name[0] == '.'
        if is_relative_name:
            input_name_as_path = Path(parent / input_name).resolve()
            if len(input_name_as_path.parents) == 1:
                raise ValueError(f'Failed to resolve input name. Unable to resolve "{input_name}" in brick "{parent.relative_to("/root")}" '
                                  'to an actual name. ')
            input_name = str(input_name_as_path.relative_to('/root'))
        input_names_resolved.append(input_name)

    if isinstance(input_or_output_names, dict):
        input_names_resolved = dict(zip(input_or_output_names.keys(), input_names_resolved))
    return input_names_resolved


@typechecked
class BrickTrainable(BrickModule):
    style: Dict[str, str] = use_default_style({'fill' :'#6D597A'})
    def __init__(self, model: nn.Module,
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str],
                 loss_output_names: Union[List[str], str] = 'none',
                 alive_stages: Union[List[Stage], str] = 'all'):
        super().__init__(model=model,
                         input_names=input_names,
                         output_names=output_names,
                         loss_output_names=loss_output_names,
                         alive_stages=alive_stages,
                         calculate_gradients=True,
                         trainable=True)

@typechecked
class BrickNotTrainable(BrickModule):
    style: Dict[str, str] = use_default_style({'fill' :'#B56576'})
    def __init__(self, model: nn.Module,
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str],
                 alive_stages: Union[List[Stage], str] = 'all',
                 calculate_gradients: bool = True):
        super().__init__(model=model,
                         input_names=input_names,
                         output_names=output_names,
                         loss_output_names='none',
                         alive_stages=alive_stages,
                         calculate_gradients=calculate_gradients,
                         trainable=False)


@typechecked
class BrickLoss(BrickModule):
    style: Dict[str, str] = use_default_style({'fill' :'#5C677D'})
    def __init__(self, model: nn.Module,
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str],
                 loss_output_names: Union[List[str], str] = 'all',
                 alive_stages: Union[List[Stage], str, None] = None,
                 ):

        alive_stages = alive_stages or [Stage.TRAIN, Stage.TEST, Stage.VALIDATION]
        super().__init__(model=model,
                         input_names=input_names,
                         output_names=output_names,
                         loss_output_names=loss_output_names,
                         alive_stages=alive_stages,
                         calculate_gradients=True,
                         trainable=True)


@typechecked
class BrickMetrics(BrickInterface, nn.Module):
    style: Dict[str, str] = use_default_style({'fill' :'#1450A3'})
    def __init__(self, metric_collection: Union[MetricCollection,  Dict[str, Metric]],
                 input_names: Union[List[str], Dict[str, str]],
                 alive_stages: Union[List[Stage], str, None] = None,
                 return_metrics: bool = False,
                 ):

        alive_stages = alive_stages or [Stage.TRAIN, Stage.TEST, Stage.VALIDATION]
        if return_metrics:
            output_names = list(metric_collection)
        else:
            output_names = []

        BrickInterface.__init__(self,
                                input_names=input_names,
                                output_names=output_names,
                                alive_stages=alive_stages)
        nn.Module.__init__(self)

        if isinstance(metric_collection, dict):
            metric_collection = MetricCollection(metric_collection)
        self.name = f'{list(metric_collection)}'
        self.return_metrics = return_metrics
        self.metrics = nn.ModuleDict({stage.name: metric_collection.clone() for stage in alive_stages})

    def forward(self, named_inputs: Dict[str, Any], stage: Stage) -> Dict[str, Any]:
        named_inputs['stage'] = stage
        metric_collection = self.metrics[stage.name]
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
                                                  calculate_gradients=False)
        if self.return_metrics:
            return output['metrics']  # Metrics in a dictionary
        else:
            assert output == {}
            return {}

    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def summarize(self, stage: Stage, reset: bool) -> Dict[str, Any]:
        metric_collection = self.metrics[stage.name]
        metrics = metric_collection.compute()
        if reset:
            metric_collection.reset()

        return metrics

    def get_module_name(self) -> str:
        return self.name

@typechecked
class BrickMetricSingle(BrickMetrics):
    def __init__(self,
                 metric: Metric,
                 input_names: Union[List[str], Dict[str, str]],
                 metric_name: Optional[str] = None,
                 alive_stages: List[Stage] | None = None,
                 return_metrics: bool = False):
        metric_name = metric_name or metric.__class__.__name__
        super().__init__(metric_collection={metric_name: metric}, input_names=input_names, alive_stages=alive_stages,
                         return_metrics=return_metrics)

@typechecked
class BrickTensorAsArrays(BrickModule):
    """
    Preferably BrickTensorsAsArray would be a only BrickInterface.
    But to have it in a brick collection (which is a inherently nn.ModuleDict) it needs to be a nn.Module/BrickModule.
    """
    style: Dict[str, str] = use_default_style({'fill' :'#5C677D'})
    def __init__(self, callable: Callable,
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str],
                 alive_stages: Union[List[Stage], str, None] = None,
                 ):

        alive_stages = alive_stages or [Stage.INFERENCE]
        super().__init__(model=self.unpack_data,
                         input_names=input_names,
                         output_names=output_names,
                         loss_output_names='none',
                         alive_stages=alive_stages,
                         calculate_gradients=False,
                         trainable=False)

    def unpack_data(self, *args, **kwargs):
        has_positional = len(args) > 0
        has_keyword_args = len(kwargs) > 0
        assert has_positional or has_keyword_args, 'No input data was provided'

        return 'blah'

    def get_module_name(self) -> str:
        return self.callable.__name__
