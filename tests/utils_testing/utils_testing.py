
from pathlib import Path
from typing import Dict

import torch
from beartype import beartype


def path_repo_root():
    return Path(__file__).parents[2]


@beartype
def assert_equal_dictionaries(d0: Dict, d1: Dict, is_close: bool = False):
    assert set(d0) == set(d1)
    for key, values in d0.items():
        if isinstance(values, torch.Tensor):
            if is_close:
                assert torch.allclose(values, d1[key])
            else:
                assert values.equal(d1[key])
        else:
            assert values == d1[key]
