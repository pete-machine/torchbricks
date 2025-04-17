import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import torch
from torch import nn
from typeguard import typechecked

from torchbricks.bricks import BrickInterface


@typechecked
class BrickCollection(nn.ModuleDict):
    def __init__(self, bricks: Dict[str, Any]) -> None:
        resolve_relative_names(bricks=bricks)
        super().__init__(convert_nested_dict_to_nested_brick_collection(bricks))

    def forward(self, named_inputs: Dict[str, Any], tags: Optional[Set[str]] = None, return_inputs: bool = True) -> Dict[str, Any]:
        gathered_named_io = dict(named_inputs)  # To keep the argument `named_inputs` unchanged

        for brick in self.values():
            if isinstance(brick, BrickCollection):
                gathered_named_io.update(brick.forward(named_inputs=gathered_named_io, tags=tags))
            else:
                if brick.run_now(tags=tags):
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

    def sub_collection(self, tags: Optional[Set[str]] = None) -> "BrickCollection":
        bricks = {}
        for name, brick in self.items():
            if isinstance(brick, BrickCollection):
                bricks[name] = brick.sub_collection(tags=tags)
            elif brick.run_now(tags=tags):
                bricks[name] = brick
        return BrickCollection(bricks)

    def to_dict(self) -> Dict[str, Any]:
        bricks_dict = {}
        for name, brick in self.items():
            if isinstance(brick, BrickCollection):
                bricks_dict[name] = brick.to_dict()
            else:
                bricks_dict[name] = brick
        return bricks_dict


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
                    "to an actual name. "
                )
            input_name = str(input_name_as_path.relative_to("/root"))
        input_names_resolved.append(input_name)

    if isinstance(input_or_output_names, dict):
        input_names_resolved = dict(zip(input_or_output_names.keys(), input_names_resolved))
    return input_names_resolved


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
def resolve_relative_names(bricks: Dict[str, Union[dict, BrickInterface]]):
    return _resolve_relative_names_recursive(bricks)
