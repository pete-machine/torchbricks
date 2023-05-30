# torch_bricks

[![codecov](https://codecov.io/gh/PeteHeine/torch_bricks/branch/main/graph/badge.svg?token=torch_bricks_token_here)](https://codecov.io/gh/PeteHeine/torch_bricks)
[![CI](https://github.com/PeteHeine/torch_bricks/actions/workflows/main.yml/badge.svg)](https://github.com/PeteHeine/torch_bricks/actions/workflows/main.yml)

Awesome torch_bricks created by PeteHeine

## Install it from PyPI

```bash
pip install torch_bricks
```

## Usage

```py
from torch_bricks import BaseClass
from torch_bricks import base_function

BaseClass().base_method()
base_function()
```

```bash
$ python -m torch_bricks
#or
$ torch_bricks
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Combines mamba and poetry
Setup is described in https://stackoverflow.com/a/71110028

I decided to combine the two to use mamba to easily manage pytorch+cuda and poetry to easily package to later easily package is as pypi library

Apart from pytorch and cuda all libraries should be install with poetry.

Consider just using mamba for installing libraries.

### Install

    conda create --name my_project_env --file conda-linux-64.lock
    conda activate my_project_env
    poetry install

### Activating the environment

    conda activate my_project_env

# Updating the environment

    # Re-generate Conda lock file(s) based on environment.yml
    conda-lock -k explicit --conda mamba -f environment.yml
    # Update Conda packages based on re-generated lock file
    mamba update --file conda-linux-64.lock
    # Update Poetry packages and re-generate poetry.lock
    poetry update
