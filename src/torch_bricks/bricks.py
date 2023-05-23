from collections import OrderedDict
from enum import Enum
from typing import Callable, Dict, Any, List, Tuple
import torch
from torch import nn
from torchmetrics import MetricCollection, Metric


class RunState(Enum):           # Gradients/backward  Eval-model    Targets
    TRAIN = 'train'             # Y                   Y             Y
    VALIDATION = 'validation'   # N                   N             Y
    TEST = 'test'               # N                   N             Y
    INFERENCE = 'inference'     # N                   N             N


def select_inputs(named_tensors: Dict[str, Any], input_names: List[str]) -> List:
    is_subset = set(input_names).issubset(named_tensors)
    assert is_subset, (f"Not all expected '{input_names=}' exists in 'named_tensors={list(named_tensors)}'. The following expected names "
                       f"'{list(set(input_names).difference(named_tensors))}' does not exist in the dictionary of 'named tensors'")
    selected_inputs = [named_tensors[name] for name in input_names]
    return selected_inputs


def name_callable_outputs(outputs: Any, output_names: List[str]) -> Dict[str, Any]:
    if not isinstance(outputs, Tuple):
        outputs = (outputs, )
    assert len(outputs) == len(output_names)
    return dict(zip(output_names, outputs))


def named_input_and_outputs_callable(callable: Callable,
                                     named_tensors: Dict[str, Any],
                                     input_names: List[str] | Dict[str, Any],
                                     output_names: List[str] | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(input_names, dict) or isinstance(output_names, dict):
        raise ValueError('Dicts are not supported yet')

    selected_inputs = select_inputs(named_tensors, input_names=input_names)
    outputs = callable(*selected_inputs)
    return name_callable_outputs(outputs=outputs, output_names=output_names)

class Brick(nn.Module):

    def forward(self, state: RunState, named_tensors: Dict[str, Any]) -> Dict[str, Any]:
        """"""
        raise ValueError(f"If you are not using 'forward' then set it to 'forward=None' in class {self}")
        return named_tensors

    def calculate_loss(self, named_tensors: Dict[str, Any]) -> Dict[str, Any]:
        """"""
        raise ValueError(f"If you are not using 'calculate_loss' then set it to 'calculate_loss=None' in class {self}")
        return {}

    def update_metrics(self, state: RunState, named_tensors: Dict[str, Any], batch_idx: int) -> None:
        """"""
        raise ValueError(f"If you are not using 'update_metrics' then set it to 'update_metrics=None' in class {self}")

    def summarize(self, state: RunState, reset: bool) -> Dict[str, Any]:
        """"""
        raise ValueError(f"If you are not using 'summarize' then set it to 'summarize=None' in class {self}")
        return {}

    def on_step(self, state: RunState, named_tensors: Dict[str, Any], batch_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        named_tensors = self.forward(state=state, named_tensors=named_tensors)
        losses = self.calculate_loss(named_tensors=named_tensors)
        named_tensors.update(losses)

        with torch.no_grad():
            self.update_metrics(state=state, named_tensors=named_tensors, batch_idx=batch_idx)
        return named_tensors, losses


class BrickCollection(Brick):
    def __init__(self, bricks: Dict[str, Brick]) -> None:
        super().__init__()
        self.bricks = nn.Sequential(OrderedDict(bricks))

    def forward(self, state: RunState, named_tensors: Dict[str, Any]) -> Dict[str, Any]:
        forward_bricks = [brick for brick in self.bricks if brick.forward is not None]
        for brick in forward_bricks:
            results = brick.forward(state=state, named_tensors=named_tensors)
            if results is None:
                results = {}
            named_tensors.update(results)
        return named_tensors

    def calculate_loss(self, named_tensors: Dict[str, Any]) -> Dict[str, Any]:
        loss_bricks = [brick for brick in self.bricks if brick.calculate_loss is not None]
        losses = {}
        for brick in loss_bricks:
            if brick.calculate_loss is not None:
                losses.update(brick.calculate_loss(named_tensors=named_tensors))
        return losses

    def update_metrics(self, state: RunState, named_tensors: Dict[str, Any], batch_idx: int) -> None:
        update_metrics_bricks = [brick for brick in self.bricks if brick.update_metrics is not None]
        for brick in update_metrics_bricks:
            brick.update_metrics(state=state, named_tensors=named_tensors, batch_idx=batch_idx)

    def summarize(self, state: RunState, reset: bool) -> Dict[str, Any]:
        update_metrics_bricks = [brick for brick in self.bricks if brick.summarize is not None]
        metrics = {}
        for brick in update_metrics_bricks:
            metrics.update(brick.summarize(state=state, reset=reset))
        return metrics

# class BrickExecutor:
#     on_step = None          # Includes forward, calculate_loss, update_metrics
#     on_epoch_end = None




class BrickTrainable(Brick):
    calculate_loss = None
    update_metrics = None
    summarize = None

    def __init__(self, model: nn.Module, input_names: list[str], output_names: list[str]):
        super().__init__()
        self.input_names = input_names
        self.output_names = output_names
        self.model = model

    def forward(self, state: RunState, named_tensors: Dict[str, Any]) -> Dict[str, Any]:
        named_tensors['state'] = state
        selected_inputs = select_inputs(named_tensors, input_names=self.input_names)
        outputs = self.model(*selected_inputs)
        return name_callable_outputs(outputs=outputs, output_names=self.output_names)


class BrickNotTrainable(Brick):
    calculate_loss = None
    update_metrics = None
    summarize = None

    def __init__(self, model: nn.Module, input_names: list[str], output_names: list[str]):
        super().__init__()
        self.input_names = input_names
        self.output_names = output_names
        self.model = model
        self.model.requires_grad_(False)

    def forward(self, state: RunState, named_tensors: Dict[str, Any]) -> Dict[str, Any]:
        named_tensors['state'] = state
        selected_inputs = select_inputs(named_tensors, input_names=self.input_names)
        with torch.no_grad():
            outputs = self.model(*tuple(selected_inputs))
        return name_callable_outputs(outputs=outputs, output_names=self.output_names)


class BrickLoss(Brick):
    forward = None
    update_metrics = None
    summarize = None

    def __init__(self, model: nn.Module, input_names: list[str], output_names: list[str]):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.output_names = output_names

    def calculate_loss(self, named_tensors: Dict[str, Any]) -> Dict[str, Any]:
        outputs = self.model(*select_inputs(named_tensors, input_names=self.input_names))
        return name_callable_outputs(outputs=outputs, output_names=self.output_names)


class BrickTorchMetric(Brick):
    forward = None
    calculate_loss = None

    def __init__(self, metric: MetricCollection | Metric, input_names: list[str], metric_name: str = ''):
        super().__init__()
        self.input_names = input_names
        val_args = {}
        train_args = {}
        test_args = {}
        if isinstance(metric, MetricCollection):
            train_args['prefix'] = self.get_metric_name(RunState.TRAIN, metric_name=metric_name)
            val_args['prefix'] = self.get_metric_name(RunState.VALIDATION, metric_name=metric_name)
            test_args['prefix'] = self.get_metric_name(RunState.TEST, metric_name=metric_name)
        self.metrics_train = metric.clone(**train_args)
        self.metrics_validation = metric.clone(**val_args)
        self.metrics_test = metric.clone(**test_args)

    def _select_metric_collection_from_split(self, state: RunState) -> MetricCollection:
        if state == RunState.TRAIN:
            return self.metrics_train
        elif state == RunState.TEST:
            return self.metrics_test
        elif state == RunState.VALIDATION:
            return self.metrics_validation
        raise TypeError('')

    @staticmethod
    def get_metric_name(state: RunState, metric_name: str) -> str:
        return f'{state.value}/{metric_name}'

    def update_metrics(self, state: RunState, named_tensors: Dict[str, Any], batch_idx: int) -> None:
        metric = self._select_metric_collection_from_split(state=state)
        named_tensors['state'] = state
        selected_inputs = select_inputs(named_tensors, input_names=self.input_names)
        metric.update(*selected_inputs)

    def summarize(self, state: RunState, reset: bool):
        metric = self._select_metric_collection_from_split(state=state)
        metrics = metric.compute()
        if reset:
            metric.reset()
        return metrics
