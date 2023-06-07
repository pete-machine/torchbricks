# TorchBricks

[![codecov](https://codecov.io/gh/PeteHeine/torchbricks/branch/main/graph/badge.svg?token=torchbricks_token_here)](https://codecov.io/gh/PeteHeine/torchbricks)
[![CI](https://github.com/PeteHeine/torchbricks/actions/workflows/main.yml/badge.svg)](https://github.com/PeteHeine/torchbricks/actions/workflows/main.yml)

TorchBricks builds pytorch models using small reuseable and decoupled parts - we call them bricks.

The concept is simple and flexible and allows you to more easily combine and swap out parts of the model (preprocessor, backbone, neck, head or post-processor), change the task of the model or extend it with multiple tasks.

## Basic use-case: Image classification
Let us see it in action:

```py
from torchbricks.bricks import BrickCollection, BrickNotTrainable, BrickTrainable, Phase

class Preprocessor(nn.Module):
    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        return raw_input/2

# Defining the model
bricks = {
    "preprocessor": BrickNotTrainable(Preprocessor(), input_names=["raw"], output_names=["processed"])
    "backbone": BrickTrainable(ResNetBackbone(), input_names=["processed"], output_names=["embedding"])
    "image_classification": BrickTrainable(ImageClassifier(), input_names=["embedding"], output_names=["logits"])
}

# Executing the model
model = BrickCollection(bricks)
outputs = model(named_tensors={"raw": input_images}, phase=Phase.TRAIN)

print(outputs.keys())
"raw", "processed", "embedding", "logits"
```

Above example defines and executes a simple DAG connecting outputs of one node to
inputs of the next node by wrapping `nn.Module`s into `BrickTrainable` and `BrickNotTrainable` bricks.
In the real-world each `nn.Module` would have arguments and stuff, but (maybe) you get the idea.

Note also that we pass in `phase=Phase.TRAIN` to explicitly specify if we are doing training, validation, test or inference. We will get back to that later.

## Basic use-case: Semantic Segmentation
After running experiments, we now realize that we also wanna do semantic segmentation.
This is how it would look like:

```py
# We can optionally keep/remove image_classification from before
bricks.pop("image_classification")

# Add upscaling and semantic segmentation nn.Modules
bricks["upscaling"] = BrickTrainable(Upscaling(), input_names=["embedding"], output_names=["embedding_upscaled"])
bricks["semantic_segmentation"] = BrickTrainable(SegmentationClassifier(), input_names=["embedding_upscaled"], output_names=["ss_logits"])

# Execute model
model = BrickCollection(bricks)
outputs = model(named_tensors={"raw": input_images}, phase=Phase.TRAIN)

print(outputs.keys())
"raw", "processed", "embedding", "embedding_upscaled", "ss_logits"
```

## Use-case: Bricks for training and evaluation
Running model inference is not all - the hard part is training and evaluation.
TorchBricks are not doing training, but an important concept of bricks is that it can also include model loss functions and metrics into single model description.

We demonstrate by an example:



By defining all trainable, non-trainable, metrics and loss functions in the model

## Install it from PyPI

```bash
pip install torchbricks
```

## Usage

```py



```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

### Install

    conda create --name torchbricks --file conda-linux-64.lock
    conda activate torchbricks
    poetry install

### Activating the environment

    conda activate torchbricks
