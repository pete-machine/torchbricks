import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Union

from torch import nn
from torchmetrics import Metric, MetricCollection
from typeguard import typechecked

from torchbricks import brick_group
from torchbricks.bricks_helper import named_input_and_outputs_callable, parse_argument_loss_output_name_indices

log = logging.getLogger(__name__)


def use_default_style(overwrites: Optional[Dict[str, str]] = None):
    overwrites = overwrites or {}
    default_style = {"stroke-width": "0px"}
    default_style.update(overwrites)
    return default_style


@typechecked
class BrickInterface(ABC):
    # 67635c,f5e26b,f1dfca,db9d38,dc9097,c4779f,c75239,84bb84,394a89,7d9dc4
    style: Dict[str, str] = use_default_style()

    def __init__(
        self, input_names: Union[List[str], Dict[str, str]], output_names: List[str], group: Union[Set[str], List[str], str]
    ) -> None:
        super().__init__()

        self.input_names = input_names
        self.output_names = output_names
        if isinstance(group, list):
            group = set(group)
        if isinstance(group, str):
            group = {group}
        self.groups: Set[str] = group

    def run_now(self, groups: Optional[Set[str]]) -> bool:
        if groups is None:
            return True
        return len(self.groups.intersection(groups)) > 0

    def __call__(self, named_inputs: Dict[str, Any], groups: Optional[Set[str]]) -> Dict[str, Any]:
        return self.forward(named_inputs=named_inputs, groups=groups)

    @abstractmethod
    def forward(self, named_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """"""

    @abstractmethod
    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """"""

    @abstractmethod
    def get_module_name(self) -> str:
        """"""

    def input_names_as_list(self) -> List[str]:
        input_names = self.input_names
        if isinstance(input_names, dict):
            input_names = list(input_names.values())
        return input_names

    def output_names_as_list(self) -> List[str]:
        return self.output_names

    def summarize(self, reset: bool) -> Dict[str, Any]:
        if hasattr(self.model, "summarize") and inspect.isfunction(self.model.summarize):
            return self.model(reset=reset)
        return {}

    def get_brick_type(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        input_names = self.input_names
        output_names = self.output_names
        groups = self.groups
        return f"{self.get_brick_type()}({self.get_module_name()}, {input_names=}, {output_names=}, {groups=})"


@typechecked
class BrickModule(nn.Module, BrickInterface):
    style: Dict[str, str] = use_default_style({"fill": "#355070"})

    def __init__(
        self,
        model: Union[nn.Module, nn.ModuleDict, Callable],
        input_names: Union[List[str], Dict[str, str]],
        output_names: List[str],
        group: Union[Set[str], List[str], str] = brick_group.MODEL,
        loss_output_names: Union[List[str], str] = "none",
        calculate_gradients: bool = True,
        trainable: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        BrickInterface.__init__(self, input_names=input_names, output_names=output_names, group=group)
        self.model = model
        self.loss_output_indices = parse_argument_loss_output_name_indices(loss_output_names, available_output_names=output_names)
        self.calculate_gradients = calculate_gradients

        if hasattr(model, "requires_grad_"):
            self.model.requires_grad_(trainable)

    def forward(self, named_inputs: Dict[str, Any]) -> Dict[str, Any]:
        named_outputs = named_input_and_outputs_callable(
            callable=self.model,
            named_inputs=named_inputs,
            input_names=self.input_names,
            output_names=self.output_names,
            calculate_gradients=self.calculate_gradients,
        )
        return named_outputs

    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        loss_output_names = [self.output_names[index] for index in self.loss_output_indices]
        named_losses = {name: loss for name, loss in named_outputs.items() if name in loss_output_names}
        return named_losses

    def get_module_name(self) -> str:
        return self.model.__class__.__name__

    def __repr__(self):  # Overwrite '__repr__' of 'BrickModule'
        return BrickInterface.__repr__(self)


@typechecked
class BrickTrainable(BrickModule):
    style: Dict[str, str] = use_default_style({"fill": "#6D597A"})

    def __init__(
        self,
        model: nn.Module,
        input_names: Union[List[str], Dict[str, str]],
        output_names: List[str],
        loss_output_names: Union[List[str], str] = "none",
        group: Union[Set[str], List[str], str] = brick_group.MODEL,
    ):
        super().__init__(
            model=model,
            input_names=input_names,
            output_names=output_names,
            loss_output_names=loss_output_names,
            group=group,
            calculate_gradients=True,
            trainable=True,
        )


@typechecked
class BrickNotTrainable(BrickModule):
    style: Dict[str, str] = use_default_style({"fill": "#B56576"})

    def __init__(
        self,
        model: nn.Module,
        input_names: Union[List[str], Dict[str, str]],
        output_names: List[str],
        group: Union[Set[str], List[str], str] = brick_group.MODEL,
        calculate_gradients: bool = True,  # To allow gradients to pass through a non-trainable model
    ):
        super().__init__(
            model=model,
            input_names=input_names,
            output_names=output_names,
            loss_output_names="none",
            group=group,
            calculate_gradients=calculate_gradients,
            trainable=False,
        )


@typechecked
class BrickLoss(BrickModule):
    style: Dict[str, str] = use_default_style({"fill": "#5C677D"})

    def __init__(
        self,
        model: nn.Module,
        input_names: Union[List[str], Dict[str, str]],
        output_names: List[str],
        loss_output_names: Union[List[str], str] = "all",
        group: Union[Set[str], List[str], str] = brick_group.LOSS,
    ):
        super().__init__(
            model=model,
            input_names=input_names,
            output_names=output_names,
            loss_output_names=loss_output_names,
            group=group,
            calculate_gradients=True,
            trainable=True,
        )


@typechecked
class BrickMetrics(BrickInterface, nn.Module):
    style: Dict[str, str] = use_default_style({"fill": "#1450A3"})

    def __init__(
        self,
        metric_collection: Union[MetricCollection, Dict[str, Metric]],
        input_names: Union[List[str], Dict[str, str]],
        group: Union[Set[str], List[str], str] = brick_group.METRIC,
        return_metrics: bool = False,
    ):
        if return_metrics:
            output_names = list(metric_collection)
        else:
            output_names = []

        BrickInterface.__init__(self, input_names=input_names, output_names=output_names, group=group)
        nn.Module.__init__(self)

        if isinstance(metric_collection, dict):
            metric_collection = MetricCollection(metric_collection)
        self.name = f"{list(metric_collection)}"
        self.return_metrics = return_metrics
        self.metrics = metric_collection.clone()

    def forward(self, named_inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.return_metrics:
            output_names = ["metrics"]
            metric_callable = self.metrics  # Return metrics as a dictionary
        else:
            output_names = []
            metric_callable = self.metrics.update  # Will not return metrics

        output = named_input_and_outputs_callable(
            callable=metric_callable,
            named_inputs=named_inputs,
            input_names=self.input_names,
            output_names=output_names,
            calculate_gradients=False,
        )
        if self.return_metrics:
            return output["metrics"]  # Metrics in a dictionary
        else:
            assert output == {}
            return {}

    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def summarize(self, reset: bool) -> Dict[str, Any]:
        metrics = self.metrics.compute()
        if reset:
            self.metrics.reset()

        return metrics

    def get_module_name(self) -> str:
        return self.name


@typechecked
class BrickMetricSingle(BrickMetrics):
    def __init__(
        self,
        metric: Metric,
        input_names: Union[List[str], Dict[str, str]],
        metric_name: Optional[str] = None,
        group: Union[Set[str], List[str], str] = brick_group.METRIC,
        return_metrics: bool = False,
    ):
        metric_name = metric_name or metric.__class__.__name__
        super().__init__(metric_collection={metric_name: metric}, input_names=input_names, group=group, return_metrics=return_metrics)
