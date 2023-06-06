# torchbricks

[![codecov](https://codecov.io/gh/PeteHeine/torchbricks/branch/main/graph/badge.svg?token=torchbricks_token_here)](https://codecov.io/gh/PeteHeine/torchbricks)
[![CI](https://github.com/PeteHeine/torchbricks/actions/workflows/main.yml/badge.svg)](https://github.com/PeteHeine/torchbricks/actions/workflows/main.yml)


## Install it from PyPI

```bash
pip install torchbricks
```

## Usage

```py
## MISSING
```

```bash
$ python -m torchbricks
#or
$ torchbricks
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
