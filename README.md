# TorchBricks

[![codecov](https://codecov.io/gh/PeteHeine/torchbricks/branch/main/graph/badge.svg?token=torchbricks_token_here)](https://codecov.io/gh/PeteHeine/torchbricks)
[![CI](https://github.com/PeteHeine/torchbricks/actions/workflows/main.yml/badge.svg)](https://github.com/PeteHeine/torchbricks/actions/workflows/main.yml)

TorchBricks builds pytorch models using small reuseable and decoupled parts - we call them bricks.

The concept is simple and flexible and allows you to more easily combine and swap out parts of the model (preprocessor, backbone, neck, head or post-processor), change the task or extend it with multiple tasks.

## Basic use-case: Image classification
Let us see it in action:

First we specify regular pytorch modules: A preprocessor, a model and a classifier


```python
import torch
from torch import nn
class PreprocessorDummy(nn.Module):
    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        return raw_input/2

class TinyModel(nn.Module):
    def __init__(self, n_channels: int, n_features: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_features, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv(tensor)

class ClassifierDummy(nn.Module):
    def __init__(self, num_classes: int, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.avgpool(tensor)))
```

We can now use torchbricks to define how the modules are connected


```python
from torchbricks.bricks import BrickCollection, BrickNotTrainable, BrickTrainable, Phase

# Defining model from bricks
bricks = {
    'preprocessor': BrickNotTrainable(PreprocessorDummy(), input_names=['raw'], output_names=['processed']),
    'backbone': BrickTrainable(TinyModel(n_channels=3, n_features=10), input_names=['processed'], output_names=['embedding']),
    'image_classifier': BrickTrainable(ClassifierDummy(num_classes=3, in_features=10), input_names=['embedding'], output_names=['logits'])
}

# Executing model
model = BrickCollection(bricks)
batch_image_example = torch.rand((1, 3, 100, 200))
outputs = model(named_inputs={'raw': batch_image_example}, phase=Phase.TRAIN)
print(outputs.keys())
```

All modules are added as entries in a regular dictionary, and for each module we 1) specify a name
2) if it is trainable or not (`BrickTrainable`/`BrickNotTrainable`) and 3) input and output names.

Finally, bricks are collected in a `BrickCollection`. A `BrickCollection` has the functionality of a
regular `nn.Module` with a `forward`-function, `to` to move to a specific device/precision,
save/loading and management of parameters etc.

Beyond that, the brick collections acts as a simple DAG, it accepts a dictionary (`named_inputs`),
executes each bricks and ensures that the output of one brick is passed to the inputs of other bricks with matching names.

Note also that we set `phase=Phase.TRAIN` to explicitly specify if we are doing training, validation, test or inference.
Specifying a phase is important, if we want a module to act in a specific way during specific phases.
We will get back to this later.

Not show here, a `BrickCollection` also supports a nested dictionary of bricks. A nested brick collections acts the same,
but it becomes easier to add and remove sub-collections bricks.

## Bag of bricks - reusable bricks modules
We provide a bag-of-bricks with commonly used `nn.Module`s

Below we create a brick collection with a real world example including a `Preprocessor`, an adaptor function to convert
torchvision resnet models into a backbone brick with no classifier and an `ImageClassifier`.


```python
from torchvision.models import resnet18

from torchbricks.bag_of_bricks import ImageClassifier, resnet_to_brick, Preprocessor

num_classes = 10
resnet_brick = resnet_to_brick(resnet=resnet18(weights=False, num_classes=num_classes),  input_name='normalized', output_name='features')
bricks = {
    'preprocessor': BrickNotTrainable(Preprocessor(), input_names=['raw'], output_names=['normalized']),
    'backbone': resnet_brick,
    'image_classifier': BrickTrainable(ImageClassifier(num_classes=num_classes, n_features=resnet_brick.model.n_backbone_features),
                                     input_names=['features'], output_names=['logits', 'probabilities', 'class_prediction']),
}
bricks
```

## Use-case: Bricks `on_step`-function for training and evaluation
In above examples, we have showed how to compose trainable and non-trainable bricks, and how a dictionary of tensors is passed
to the forward function... But TorchBricks goes beyond that.

An important feature of a brick collection is the `on_step`-function to also calculate metrics and losses.

We will extend the example from before:


```python
from torchbricks.bricks import BrickLoss, BrickTorchMetric
from torchmetrics.classification import MulticlassAccuracy


bricks["accuracy"] = BrickTorchMetric(MulticlassAccuracy(num_classes=num_classes), input_names=['class_prediction', 'targets'])
bricks["loss"] = BrickLoss(model=nn.CrossEntropyLoss(), input_names=['logits', 'targets'], output_names=['loss_ce'])


# We can still run the forward-pass as before - Note: The forward call does not require 'targets'
model = BrickCollection(bricks)
batch_image_example = torch.rand((1, 3, 100, 200))
outputs = model(named_inputs={"raw": batch_image_example}, phase=Phase.TRAIN)

# Example of running `on_step`. Note: `on_step` requires `targets` to calculate metrics and loss.
named_inputs = {"raw": batch_image_example, "targets": torch.ones((1), dtype=torch.int64)}
named_outputs, losses = model.on_step(phase=Phase.TRAIN, named_inputs=named_inputs, batch_idx=0)
named_outputs, losses = model.on_step(phase=Phase.TRAIN, named_inputs=named_inputs, batch_idx=1)
named_outputs, losses = model.on_step(phase=Phase.TRAIN, named_inputs=named_inputs, batch_idx=2)
metrics = model.summarize(phase=Phase.TRAIN, reset=True)
```

## Basic use-case: Semantic Segmentation
After running experiments, we now realize that we also wanna do semantic segmentation.
This is how it would look like:


```python
missing_implementation = True
if missing_implementation:
    print("MISSING")
else:
    # We can optionally keep/remove image_classification from before
    bricks.pop("image_classifier")

    # Add upscaling and semantic segmentation nn.Modules
    bricks["upscaling"] = BrickTrainable(Upscaling(), input_names=["embedding"], output_names=["embedding_upscaled"])
    bricks["semantic_segmentation"] = BrickTrainable(SegmentationClassifier(), input_names=["embedding_upscaled"], output_names=["ss_logits"])

    # Executing model
    model = BrickCollection(bricks)
    batch_image_example = torch.rand((1, 3, 100, 200))
    outputs = model(named_inputs={"raw": batch_image_example}, phase=Phase.TRAIN)

    print(outputs.keys())
```

By wrapping both core model computations, metrics and loss functions into a single brick collection, we can more easily swap between
running model experiments in notebooks, trainings

We provide a `forward` function to easily run model inference without targets and an `on_step` function
to easily get metrics and losses in both

## Use-case: Training with a collections of bricks
We like and love pytorch-lightning! We can avoid writing the easy-to-get-wrong training loop, write validation/test scrips, it create
logs, ensures training is done efficiently on any device (CPU, GPU, TPU), on multiple devices with reduced precision and much more.

But with pytorch-lightning you need to specify a LightningModule and I find myself hiding the important stuff in the class
and using multiple levels of inheritance. It can make your code unnecessarily complicated, hard to read and hard to reuse.
It may also require some heavy refactoring changing to a new task or switching to multiple tasks.

With a brick collection you should rarely change or inherit your lightning module, instead you inject the model, metrics and loss functions
into a lightning module. Changes to preprocessor, backbone, necks, heads, metrics and losses are done on the outside
and injected into the lightning module.

Below is an example of how you could inject our brick collection into our custom `LightningBrickCollection`.
The brick collection can be image classification, semantic segmentation, object detection or all of them at the same time.



```python
create_opimtizer_func = partial(torch.optim.SGD, lr=0.05, momentum=0.9, weight_decay=5e-4)
bricks_lightning_module = LightningBrickCollection(path_experiments=path_experiments,
                                                   experiment_name=experiment_name,
                                                   brick_collection=brick_collection,
                                                   create_optimizer_func=create_opimtizer_func)

logger = WandbLogger(name=experiment_name, project=PROJECT)
trainer = Trainer(accelerator=args.accelerator, logger=logger, max_epochs=args.max_epochs)
trainer.fit(bricks_lightning_module,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader())
trainer.test(bricks_lightning_module, datamodule=data_module)
```

## Nested brick collections
It can handle nested brick collections and nested dictionary of bricks.

MISSING

## TorchMetric.MetricCollection

MISSING

## Why should I explicitly set the train, val or test phase

MISSING

##

## What are we missing?


- [ ] Proper `LightningBrickCollection` for other people to use
- [ ] Collection of helper modules. Preprocessors, Backbones, Necks/Upsamplers, ImageClassification, SemanticSegmentation, ObjectDetection
  - [ ] All the modules in the README should be easy to import as actually modules.
  - [ ] Make common brick collections: BricksImageClassification, BricksSegmentation, BricksPointDetection, BricksObjectDetection
- [ ] Support preparing data in the dataloader?
- [ ] Make common Visualizations with pillow - not opencv to not blow up the required dependencies. ImageClassification, Segmentation, ObjectDetection
- [ ] Make an export to onnx function and add it to the README.md
- [ ] Proper handling of train, val and test. What to do with gradients, nn.Module parameters and internal eval/train state
- [ ] Consider: If train, val and test phase has no impact on bricks, it should be passed as a regular named input.
- [x] Minor: BrickCollections supports passing a dictionary with BrickCollections. But we should also convert a nested dictionary into a nested brick collections
- [x] Minor: Currently, `input_names` and `output_names` support positional arguments, but we should also support keyword arguments.
- [x] Minor: Make Brick an abstract class
- [x] Convert torchvision resnet models to only a backbone brick.
- [ ] Make readme a notebook
- [ ] Ensure that all examples in the `README.md` are working with easy to use modules.
- [ ] Test: Make it optional if gradients can be passed through NonTrainableBrick without weights being optimized


## How does it really work?
????

## Install it from PyPI

```bash
pip install torchbricks
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

### Install

    conda create --name torchbricks --file conda-linux-64.lock
    conda activate torchbricks
    poetry install

### Activating the environment

    conda activate torchbricks
