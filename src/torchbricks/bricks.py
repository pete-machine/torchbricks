import inspect
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

import torch
from torch import nn
from torchmetrics import Metric, MetricCollection
from typeguard import typechecked

from torchbricks import brick_group
from torchbricks.bricks_helper import named_input_and_outputs_callable

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
def parse_argument_loss_output_name_indices(loss_output_names: Union[List[str], str], available_output_names: List[str]) -> List[int]:
    if loss_output_names == "all":
        loss_output_names = available_output_names
    elif loss_output_names == "none":
        loss_output_names = []

    assert isinstance(loss_output_names, List), f"`loss_output_names` should be `all`, `none` or a list of strings. {loss_output_names=}"
    assert set(loss_output_names).issubset(available_output_names), (
        f"One or more {loss_output_names=} is not an " "`output_names` of brick {output_names=}"
    )
    return [available_output_names.index(loss_output_name) for loss_output_name in loss_output_names]


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
class BrickCollection(nn.ModuleDict):
    def __init__(self, bricks: Dict[str, Any]) -> None:
        resolve_relative_names(bricks=bricks)
        super().__init__(convert_nested_dict_to_nested_brick_collection(bricks))

    def forward(self, named_inputs: Dict[str, Any], groups: Optional[Set[str]] = None, return_inputs: bool = True) -> Dict[str, Any]:
        gathered_named_io = dict(named_inputs)  # To keep the argument `named_inputs` unchanged

        for brick in self.values():
            if isinstance(brick, BrickCollection):
                gathered_named_io.update(brick.forward(named_inputs=gathered_named_io, groups=groups))
            else:
                if brick.run_now(groups=groups):
                    results = brick.forward(named_inputs=gathered_named_io)
                    gathered_named_io.update(results)

        if not return_inputs:
            [gathered_named_io.pop(name_input) for name_input in named_inputs]
        return gathered_named_io

    def extract_losses(self, named_outputs: Dict[str, Any]) -> Dict[str, Any]:
        named_losses = {}
        for brick in self.values():
            named_losses.update(brick.extract_losses(named_outputs=named_outputs))
        return named_losses

    @typechecked
    def save_bricks(self, path_model_folder: Path, exist_ok: bool = False):
        for name, brick in self.items():
            if isinstance(brick, BrickCollection):
                brick.save_bricks(path_model_folder=path_model_folder / name, exist_ok=exist_ok)
            else:
                named_parameters = brick.state_dict()
                if len(named_parameters) == 0:
                    continue
                path_model_folder.mkdir(parents=True, exist_ok=exist_ok)
                path_folder_brick = str(path_model_folder / name) + ".pt"
                if not exist_ok and Path(path_folder_brick).exists():
                    raise FileExistsError(f"File exists: {path_folder_brick}")
                torch.save(brick.state_dict(), path_folder_brick)

    @typechecked
    def load_bricks(self, path_model_folder: Path):
        if not path_model_folder.exists():
            raise FileNotFoundError(f"Model directory not found: {path_model_folder}")
        if not path_model_folder.is_dir():
            raise NotADirectoryError(f"Model path is not a directory: {path_model_folder}")
        self._load_from_folder(path_base=path_model_folder, nesting=None)

    @typechecked
    def _load_from_folder(self, path_base: Path, nesting: Optional[Path]) -> None:
        nesting = nesting or Path("")
        path_model_folder = path_base / nesting
        for name, brick in self.items():
            if isinstance(brick, BrickCollection):
                brick._load_from_folder(path_base=path_base, nesting=nesting / name)
            else:
                named_parameters = brick.state_dict()
                if len(named_parameters) == 0:
                    continue
                path_folder_brick = (path_model_folder / name).with_suffix(".pt")
                if not Path(path_folder_brick).exists():
                    warnings.warn(f"Brick '{nesting / name}' has no matching weight file and is initialized from scratch. ", stacklevel=1)
                    continue
                brick.load_state_dict(torch.load(path_folder_brick))

    def summarize(self, reset: bool) -> Dict[str, Any]:
        metrics = {}
        for brick in self.values():
            metrics.update(brick.summarize(reset=reset))
        return metrics


@typechecked
def convert_nested_dict_to_nested_brick_collection(bricks: Dict[str, Union[BrickInterface, Dict]], level=0):
    converted_bricks = {}
    for name, brick in bricks.items():
        if isinstance(brick, dict):
            converted_bricks[name] = convert_nested_dict_to_nested_brick_collection(brick, level=level + 1)
        else:
            converted_bricks[name] = brick

    if level == 0:
        return converted_bricks
    else:
        return BrickCollection(converted_bricks)


@typechecked
def resolve_relative_names(bricks: Dict[str, Union[dict, BrickInterface]]):
    return _resolve_relative_names_recursive(bricks)


@typechecked
def _resolve_relative_names_recursive(
    bricks: Union[Dict[str, Union[dict, BrickInterface]], BrickCollection], parent: Optional[Path] = None
):
    parent = parent or Path("/root")
    for brick_name, brick_or_bricks in bricks.items():
        if isinstance(brick_or_bricks, (dict, BrickCollection)):
            _resolve_relative_names_recursive(bricks=brick_or_bricks, parent=parent / brick_name)
        elif isinstance(brick_or_bricks, BrickInterface):
            brick_or_bricks.input_names = _resolve_input_or_output_names(brick_or_bricks.input_names, parent)
            brick_or_bricks.output_names = _resolve_input_or_output_names(brick_or_bricks.output_names, parent)
        else:
            raise ValueError("")

    return bricks


@typechecked
def _resolve_input_or_output_names(
    input_or_output_names: Union[List[str], Dict[str, str]], parent: Path
) -> Union[List[str], Dict[str, str]]:
    if isinstance(input_or_output_names, dict):
        data_names = input_or_output_names.values()
    else:
        data_names = list(input_or_output_names)
    input_names_resolved = []
    for input_name in data_names:
        is_relative_name = input_name[0] == "."
        if is_relative_name:
            input_name_as_path = Path(parent / input_name).resolve()
            if len(input_name_as_path.parents) == 1:
                raise ValueError(
                    f'Failed to resolve input name. Unable to resolve "{input_name}" in brick "{parent.relative_to("/root")}" '
                    'to an actual name. '
                )
            input_name = str(input_name_as_path.relative_to("/root"))
        input_names_resolved.append(input_name)

    if isinstance(input_or_output_names, dict):
        input_names_resolved = dict(zip(input_or_output_names.keys(), input_names_resolved))
    return input_names_resolved


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
        calculate_gradients: bool = True,
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
