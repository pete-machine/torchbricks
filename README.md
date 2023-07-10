<!--

---
jupyter:
  jupytext:
    hide_notebook_metadata: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: torchbricks
    language: python
    name: python3
---

-->

# TorchBricks

[![codecov](https://codecov.io/gh/PeteHeine/torchbricks/branch/main/graph/badge.svg?token=torchbricks_token_here)](https://codecov.io/gh/PeteHeine/torchbricks)
[![CI](https://github.com/PeteHeine/torchbricks/actions/workflows/main.yml/badge.svg)](https://github.com/PeteHeine/torchbricks/actions/workflows/main.yml)

TorchBricks builds pytorch models using small reuseable and decoupled parts - we call them bricks.

The concept is simple and flexible and allows you to more easily combine, add more or swap out parts of the model (preprocessor, backbone, neck, head or post-processor), change the task or extend it with multiple tasks.


<!-- #region -->

## Install it with pip

```bash
pip install torchbricks
```
<!-- #endregion -->

## Bricks by examples

First we specify regular pytorch modules: A preprocessor, a model and a classifier

```python
from typing import Tuple
import torch
from torch import nn
class PreprocessorDummy(nn.Module):
    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        return raw_input/2

class TinyModel(nn.Module):
    def __init__(self, n_channels: int, n_features: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_features, kernel_size=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv(tensor)

class ClassifierDummy(nn.Module):
    def __init__(self, num_classes: int, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(torch.flatten(self.avgpool(tensor), start_dim = 1))
        softmaxed = self.softmax(logits)
        return logits, softmaxed
```


## Concept 1: Bricks are connected
Using named input and output names, we specify howmodules are connected.

```python
from torchbricks.bricks import BrickCollection, BrickModule
from torchbricks.bricks import Stage

# Defining model from bricks
bricks = {
    'preprocessor': BrickModule(PreprocessorDummy(), input_names=['raw'], output_names=['processed']),
    'backbone': BrickModule(TinyModel(n_channels=3, n_features=10), input_names=['processed'], output_names=['embedding']),
    'head': BrickModule(ClassifierDummy(num_classes=3, in_features=10), input_names=['embedding'], output_names=['logits', "softmaxed"]),
}
```

<!-- #region -->
All modules are added as entries in a regular dictionary and for each module we provide a name (dictionary key) and 
input and output names. The number of input and output names should match the actually number of input and output names 
for each function. 

The `preprocessor` uses a `raw` input tensor and outputs a `processed` tensor. The `backbone` uses a `processed` tensor and returns 
the `embedding` tensor. Finally, the `head` uses `embedding`s and outputs both model `logits` and `softmaxed` tensors. 


Bricks are then passed to what we call a `BrickCollection` for executing the bricks. 
<!-- #endregion -->

```python
brick_collection = BrickCollection(bricks)
batch_images = torch.rand((2, 3, 100, 200))
outputs = brick_collection(named_inputs={'raw': batch_images}, stage=Stage.TRAIN)
print(outputs.keys())
```

Running our models as a brick collection has the following advantages:

- A brick collection act as a regular `nn.Module` with all the familiar features: a `forward`-function, a `to`-function to move 
  to a specific device/precision, you can save/load a model, management of parameters, onnx exportable etc. 
- A brick collection is also a simple DAG, it accepts a dictionary (`named_inputs`), 
executes each bricks and ensures that the outputs are passed to the inputs of other bricks with matching names. 
Structuring the model as a DAG, makes it easy to add/remove outputs for a given module during development, add new modules to the
collection and build completely new models from reusable parts. 
- A brick collection is actually a dictionary (`nn.DictModule`). Allowing you to access, pop and update the 
  collection easily as a regular dictionary. It can to handle nested dictionary, allowing groups of bricks to be added/removed easily. 

Note also that we set `stage=stage.TRAIN` to explicitly specify if we are doing training, validation, test or inference.
Specifying a stage is important, if we want a module to act in a specific way during specific stages.
We will get back to this in the next section.



## Concept 2: Bricks can be dead or alive
The second concept is to specify when bricks are alive - meaning we specify at which stages (train, test, validation, inference and export) 
a brick is executed. For other stage the brick will play dead - do nothing / return empty dictionary. 

In above example this is not particular interesting - because preprocessor, backbone model and head would typically by alive in all stages. 

So we will demonstrate by adding a loss function and specifying `alive_stages` for each brick.

```python
bricks = {
    'preprocessor': BrickModule(PreprocessorDummy(), input_names=['raw'], output_names=['processed'], alive_stages="all"),
    'backbone': BrickModule(TinyModel(n_channels=3, n_features=10), input_names=['processed'], output_names=['embedding'], alive_stages="all"),
    'head': BrickModule(ClassifierDummy(num_classes=3, in_features=10), input_names=['embedding'], output_names=['logits', 'softmaxed'], 
                                    alive_stages="all"),
    'loss': BrickModule(model=nn.CrossEntropyLoss(), input_names=['logits', 'targets'], output_names=['loss_ce'], 
                        alive_stages=[Stage.TRAIN, Stage.VALIDATION, Stage.TEST], loss_output_names="all")
}
```

We set `preprocessor`, `backbone` and `head` to be alive on all stages `alive_stages="all"` - this is the default behavior and
similar to before. 

For `loss` we set `alive_stages=[Stage.TRAIN, Stage.VALIDATION, Stage.TEST]`. 

Also note that the `loss` brick requires an additional input called `targets`.



```python
brick_collection = BrickCollection(bricks)
batch_images = torch.rand((1, 3, 100, 200))
outputs_without_loss = brick_collection(named_inputs={'raw': batch_images}, stage=Stage.INFERENCE)
outputs_with_loss = brick_collection(named_inputs={'raw': batch_images, "targets": torch.ones((1,3))}, stage=Stage.TRAIN)
```


With `stage=Stage.INFERENCE`, the brick collection will act as before - the loss will not be executed and `targets` will not be required. 

With `stage=Stage.TRAIN`, the brick collection requires `targets` and returns also the loss.


## Concept 1 and 2 in a "real" use case
We now want to demonstrate brick in a "real" use case. 

```python
import torchvision
from torchbricks.bag_of_bricks import ImageClassifier, Preprocessor, resnet_to_brick
from torchbricks.bricks import BrickLoss, BrickNotTrainable, BrickTrainable, BrickMetricSingle
from torchmetrics.classification import MulticlassAccuracy

num_classes = 10
resnet = torchvision.models.resnet18(weights=None, num_classes=num_classes)
resnet_brick = resnet_to_brick(resnet=resnet,  input_name='normalized', output_name='features')

bricks = {
    'preprocessor': BrickNotTrainable(Preprocessor(), input_names=['raw'], output_names=['normalized']),
    'backbone': resnet_brick,
    'head': BrickTrainable(ImageClassifier(num_classes=num_classes, n_features=resnet_brick.model.n_backbone_features),
                                     input_names=['features'], output_names=['logits', 'probabilities', 'class_prediction']),
    'accuracy': BrickMetricSingle(MulticlassAccuracy(num_classes=num_classes), input_names=['class_prediction', 'targets']),
    'loss': BrickLoss(model=nn.CrossEntropyLoss(), input_names=['logits', 'targets'], output_names=['loss_ce'])
}

brick_collection = BrickCollection(bricks)
named_inputs = {"raw": batch_images, "targets": torch.ones((1), dtype=torch.int64)}
named_outputs = brick_collection(named_inputs=named_inputs, stage=Stage.TRAIN)
named_outputs = brick_collection(named_inputs=named_inputs, stage=Stage.TRAIN)
named_outputs = brick_collection(named_inputs=named_inputs, stage=Stage.TRAIN)
named_outputs = brick_collection(named_inputs=named_inputs, stage=Stage.TRAIN)
metrics = brick_collection.summarize(stage=Stage.TRAIN, reset=True)
print(f"{metrics=}, {named_outputs.keys()=}")
```


In this example we do not use `BrickModule` to build our collection - you can do that -
but instead we recommend using our pre-configured brick modules (`BrickLoss`, `BrickNotTrainable`, `BrickTrainable`, 
`BrickMetricSingle` and `BrickCollection`) to both ensure sensible defaults and to show the intend of each brick. 

For metrics, we rely on the [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/) library and passes either a single 
metric (`torchmetrics.Metric`) to `BrickMetricSingle` or a collection of metrics (`torchmetrics.MetricCollection`) to `BrickMetricMultiple`.

For multiple metrics use always `BrickMetricMultiple` with `torchmetrics.MetricCollection`. 

On each `forward`, we calculate model outputs, losses and metrics for each batch. Metrics are aggregated internally in `BrickMetricSingle` 
and only returned with the `summarize`-call. We set `reset=True` to reset metric aggregation. 


### Bag of bricks - reusable bricks modules
Note also in above example we use bag-of-bricks to import commonly used `nn.Module`s 

This includes a `Preprocessor`, `ImageClassifier` and `resnet_to_brick` to convert torchvision resnet models into a backbone brick 
(without a classifier).


## Bricks motivation (to be continued)

The main motivation:

- Avoid modules within modules within modules to created models that are combined. 
- Not flexible. It is possible to make the first encode/decode model... But adding a preprocessor, swapping out a backbone,
  adding additional heads or necks and sharing computations will typically not be easy. I ended up creating multiple modules that are
  called within other modules... All head/modules pass dictionaries between modules. 
- Typically not very reusable. 

Including metrics and losses with the model. 
- Model, metrics and losses are connected. If we want to add an additional head to a model - we should also add losses and metrics. 
- The typical distinction between `encode`  / `decoder` becomes to limited... Multiple decoders might share a `neck`.


## Use-case: Training with a collections of bricks

By packing model modules, metrics and loss-functions into a brick collection and providing a `on_step`-function, we can more easily 
inject any desired brick collection into your custom trainer without doing modifications to trainer.

### Use-case: Training with pytorch-lightning trainer
I like and love pytorch-lightning! We can avoid writing the easy-to-get-wrong training loop, write validation/test scrips.

Pytorch lightning will create logs, ensures training is done efficiently on any device (CPU, GPU, TPU), on multiple/distributed devices 
with reduced precision and much more.

However, one issue I found myself having when wanting to extend my custom pytorch-lightning module (`LightningModule`) is that it forces an
object oriented style with multiple levels of inheritance. This is not necessarily bad, but it makes it hard to reuse 
code across projects and generally made the code complicated. 

With a brick collection you should rarely change or inherit your lightning module, instead you inject the model, metrics and loss functions
into a lightning module. Changes to preprocessor, backbone, necks, heads, metrics and losses are done on the outside
and injected into the lightning module. 

Below is an example of how you could inject a brick collection into with pytorch-lightning. 
We have created `LightningBrickCollection` ([available here](https://github.com/PeteHeine/torchbricks/blob/main/scripts/lightning_module.py)) 
as an example for you to use. 


```python
from functools import partial
from pathlib import Path

import torchvision
import pytorch_lightning as pl
from utils_testing.lightning_module import LightningBrickCollection
from utils_testing.datamodule_cifar10 import CIFAR10DataModule

experiment_name="CIFAR10"
transform = torchvision.transforms.ToTensor()
data_module = CIFAR10DataModule(data_dir='data', batch_size=5, num_workers=12, test_transforms=transform, train_transforms=transform)
create_opimtizer_func = partial(torch.optim.SGD, lr=0.05, momentum=0.9, weight_decay=5e-4)
bricks_lightning_module = LightningBrickCollection(path_experiments=Path("build") / "experiments",
                                                   experiment_name=None,
                                                   brick_collection=brick_collection,
                                                   create_optimizers_func=create_opimtizer_func)

trainer = pl.Trainer(accelerator="cpu", max_epochs=1, limit_train_batches=2, limit_val_batches=2, limit_test_batches=2)
# To train and test model
trainer.fit(bricks_lightning_module, datamodule=data_module)
trainer.test(bricks_lightning_module, datamodule=data_module)
```

By wrapping both core model computations, metrics and loss functions into a single brick collection, we can more easily swap between
running model experiments in notebooks, trainings

We provide a `forward` function to easily run model inference without targets and an `on_step` function
to easily get metrics and losses in both


## Nested brick collections
Not shown here, a `BrickCollection` also supports a nested dictionary of bricks. A nested brick collections acts the same, 
but it becomes easier to add and remove sub-collections bricks. 

MISSING





## TorchMetric.MetricCollection

MISSING






## Why should I explicitly set the train, val or test stage

MISSING



<!-- #region -->
##

## What are we missing?


- [x] ~~Proper~~ Added a link to `LightningBrickCollection` for other people to use
- [x] Minor: BrickCollections supports passing a dictionary with BrickCollections. But we should also convert a nested dictionary into a nested brick collections
- [x] Minor: Currently, `input_names` and `output_names` support positional arguments, but we should also support keyword arguments.
- [x] Minor: Make Brick an abstract class
- [x] Convert torchvision resnet models to only a backbone brick.
- [x] Make readme a notebook
- [x] Automatically convert jupyter notebook to `README.md`
- [x] Remove README.md header
- [x] Make an export to onnx function 
- [x] Make it optional if gradients can be passed through NonTrainableBrick without weights being optimized
- [x] Refactor Metrics: Create BrickMetricCollection and BrickSingleMetric and create flag to return metrics.
- [x] Make brick base class with input_names, output_names and alive_stages - inherit this from other bricks. 
  - Pros: We might include other non-torch modules later. 
  - Do not necessarily pass a stage-object. Consider also passing it as a string so it can be handled correctly with scripting. 
- [x] Update README.md to match the new bricks. 
  - [x] Start with basic bricks example. 
  - [x] Use loss-function to show that stage decided on what is being executed. 
  - [x] Introduce metrics by it-self in another example
- [x] Ensure that all examples in the `README.md` are working with easy to use modules. 
- [ ] Add onnx export example to the README.md
- [ ] Make DAG like functionality to check if a inputs and outputs works for all model stages.
- [ ] Use pymy, pyright or pyre to do static code checks. 
- [ ] Decide: Add stage as an internal state and not in the forward pass:
  - Minor Pros: Tracing (to get onnx model) requires only torch.Tensors only as input - we avoid making an adapter class. 
  - Minor Cons: State gets hidden away - implicit instead of explicit.
  - Minor Pros: Similar to eval/training 
- [ ] Collection of helper modules. Preprocessors, Backbones, Necks/Upsamplers, ImageClassification, SemanticSegmentation, ObjectDetection
  - [ ] All the modules in the README should be easy to import as actually modules.
  - [ ] Make common brick collections: BricksImageClassification, BricksSegmentation, BricksPointDetection, BricksObjectDetection
- [ ] Support preparing data in the dataloader?
- [ ] Make common Visualizations with pillow - not opencv to not blow up the required dependencies. ImageClassification, Segmentation, ObjectDetection

## How does it really work?
????



## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

### Install

    conda create --name torchbricks --file conda-linux-64.lock
    conda activate torchbricks
    poetry install

### Activating the environment

    conda activate torchbricks

<!-- #endregion -->


