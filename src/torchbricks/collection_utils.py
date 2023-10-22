

from collections.abc import MutableMapping
from typing import Any, Dict, Optional

from torchbricks.bricks import BrickCollection


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, BrickCollection):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
    return dict(_flatten_dict_gen(d, parent_key, sep))


def unflatten(d: Dict[str, Any], base: Optional[Dict[str, Any]] = None, sep: str = '/') -> Dict[str, Any]:
    """
    Convert any keys containing dotted paths to nested dicts
    https://stackoverflow.com/a/55545369
    """
    if base is None:
        base = {}

    for key, value in d.items():
        root = base

        if sep in key:
            *parts, key = key.split(sep)

            for part in parts:
                root.setdefault(part, {})
                root = root[part]

        if isinstance(value, dict):
            value = unflatten(value, root.get(key, {}), sep=sep)

        root[key] = value

    return base
