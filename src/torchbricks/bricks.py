from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Tuple, Union
import torch
from torch import nn
from torchmetrics import MetricCollection, Metric

from torchbricks.bricks_helper import named_input_and_outputs_callable, select_inputs_by_name


class Phase(Enum):              # Gradients/backward  Eval-model    Targets
    TRAIN = 'train'             # Y                   Y             Y
    VALIDATION = 'validation'   # N                   N             Y
    TEST = 'test'               # N                   N             Y
    INFERENCE = 'inference'     # N                   N             N


class Brick(nn.Module, ABC):

    @abstractmethod
    def forward(self, phase: Phase, named_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """"""
        raise ValueError(f"If you are not using 'forward' then set it to 'forward=None' in class {self}")

    @abstractmethod
    def calculate_loss(self, named_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """"""
        raise ValueError(f"If you are not using 'calculate_loss' then set it to 'calculate_loss=None' in class {self}")

    @abstractmethod
    def update_metrics(self, phase: Phase, named_inputs: Dict[str, Any], batch_idx: int) -> None:
        """"""
        raise ValueError(f"If you are not using 'update_metrics' then set it to 'update_metrics=None' in class {self}")

    @abstractmethod
    def summarize(self, phase: Phase, reset: bool) -> Dict[str, Any]:
        """"""
        raise ValueError(f"If you are not using 'summarize' then set it to 'summarize=None' in class {self}")

    def on_step(self, phase: Phase, named_inputs: Dict[str, Any], batch_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        named_inputs = self.forward(phase=phase, named_inputs=named_inputs)
        losses = self.calculate_loss(named_inputs=named_inputs)
        named_inputs.update(losses)

        with torch.no_grad():
            self.update_metrics(phase=phase, named_inputs=named_inputs, batch_idx=batch_idx)
        return named_inputs, losses


def convert_nested_dict_to_nested_brick_collection(bricks: Dict[str, Union[Brick, Dict]], level=0):
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


class BrickCollection(nn.ModuleDict, Brick):  # Note BrickCollection is inherently ModuleDict and acts as a dictionary of modules
    def __init__(self, bricks: Dict[str, Brick]) -> None:
        super().__init__(convert_nested_dict_to_nested_brick_collection(bricks))

    def forward(self, phase: Phase, named_inputs: Dict[str, Any]) -> Dict[str, Any]:
        named_inputs = dict(named_inputs)  # To keep `named_inputs` unchanged
        forward_bricks = [brick for brick in self.values() if brick.forward is not None]
        for brick in forward_bricks:
            results = brick.forward(phase=phase, named_inputs=named_inputs)
            if results is None:
                results = {}
            named_inputs.update(results)
        return named_inputs

    def calculate_loss(self, named_inputs: Dict[str, Any]) -> Dict[str, Any]:
        loss_bricks = [brick for brick in self.values() if brick.calculate_loss is not None]
        losses = {}
        for brick in loss_bricks:
            if brick.calculate_loss is not None:
                losses.update(brick.calculate_loss(named_inputs=named_inputs))
        return losses

    def update_metrics(self, phase: Phase, named_inputs: Dict[str, Any], batch_idx: int) -> None:
        update_metrics_bricks = [brick for brick in self.values() if brick.update_metrics is not None]
        for brick in update_metrics_bricks:
            brick.update_metrics(phase=phase, named_inputs=named_inputs, batch_idx=batch_idx)

    def summarize(self, phase: Phase, reset: bool) -> Dict[str, Any]:
        update_metrics_bricks = [brick for brick in self.values() if brick.summarize is not None]
        metrics = {}
        for brick in update_metrics_bricks:
            metrics.update(brick.summarize(phase=phase, reset=reset))
        return metrics


class BrickTrainable(Brick):
    calculate_loss = None
    update_metrics = None
    summarize = None

    def __init__(self, model: nn.Module,
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str]):
        super().__init__()
        self.input_names = input_names
        self.output_names = output_names
        self.model = model

    def forward(self, phase: Phase, named_inputs: Dict[str, Any]) -> Dict[str, Any]:
        named_inputs['phase'] = phase
        return named_input_and_outputs_callable(callable=self.model,
                                                named_inputs=named_inputs,
                                                input_names=self.input_names,
                                                output_names=self.output_names)


class BrickNotTrainable(Brick):
    calculate_loss = None
    update_metrics = None
    summarize = None

    def __init__(self, model: nn.Module,
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str]):
        super().__init__()
        self.input_names = input_names
        self.output_names = output_names
        self.model = model
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, phase: Phase, named_inputs: Dict[str, Any]) -> Dict[str, Any]:
        named_inputs['phase'] = phase
        return named_input_and_outputs_callable(callable=self.model,
                                                named_inputs=named_inputs,
                                                input_names=self.input_names,
                                                output_names=self.output_names)



class BrickLoss(Brick):
    forward = None
    update_metrics = None
    summarize = None

    def __init__(self, model: nn.Module,
                 input_names: Union[List[str], Dict[str, str]],
                 output_names: List[str]):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.output_names = output_names

    def calculate_loss(self, named_inputs: Dict[str, Any]) -> Dict[str, Any]:
        return named_input_and_outputs_callable(callable=self.model,
                                                named_inputs=named_inputs,
                                                input_names=self.input_names,
                                                output_names=self.output_names)


class BrickTorchMetric(Brick):
    forward = None
    calculate_loss = None

    def __init__(self, metric: Union[MetricCollection, Metric],
                 input_names: Union[List[str], Dict[str, str]],
                 metric_name: str = ''):
        super().__init__()
        self.input_names = input_names
        val_args = {}
        train_args = {}
        test_args = {}
        if isinstance(metric, MetricCollection):
            train_args['prefix'] = self.get_metric_name(Phase.TRAIN, metric_name=metric_name)
            val_args['prefix'] = self.get_metric_name(Phase.VALIDATION, metric_name=metric_name)
            test_args['prefix'] = self.get_metric_name(Phase.TEST, metric_name=metric_name)
        self.metrics_train = metric.clone(**train_args)
        self.metrics_validation = metric.clone(**val_args)
        self.metrics_test = metric.clone(**test_args)

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

    def update_metrics(self,
                       phase: Phase,
                       named_inputs: Union[List[str], Dict[str, str]],
                       batch_idx: int) -> None:
        metric = self._select_metric_collection_from_split(phase=phase)
        named_inputs['phase'] = phase
        selected_inputs = select_inputs_by_name(named_inputs, input_names=self.input_names)
        metric.update(*selected_inputs)

    def summarize(self, phase: Phase, reset: bool):
        metric = self._select_metric_collection_from_split(phase=phase)
        metrics = metric.compute()
        if reset:
            metric.reset()
        return metrics
