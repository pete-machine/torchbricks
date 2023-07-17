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

<!-- #region -->

## Install it with pip

```bash
pip install torchbricks
```
<!-- #endregion -->

## Bricks by example

First we specify some regular model modules: A preprocessor, a model and a classifier

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
        return logits, self.softmax(logits)
```


## Concept 1: Bricks are connected
Using input and output names, we specify how modules are connected.

```python
from torchbricks.bricks import BrickCollection, BrickTrainable, BrickNotTrainable, BrickLoss
from torchbricks.bricks import Stage

bricks = {
    'preprocessor': BrickNotTrainable(PreprocessorDummy(), 
                                      input_names=['raw'], 
                                      output_names=['processed']),
    'backbone': BrickTrainable(TinyModel(n_channels=3, n_features=10), 
                               input_names=['processed'], 
                               output_names=['embedding']),
    'head': BrickTrainable(ClassifierDummy(num_classes=3, in_features=10), 
                           input_names=['embedding'], 
                           output_names=['logits', "softmaxed"]),
}
brick_collection = BrickCollection(bricks)
print(brick_collection)
# BrickCollection(
#   (preprocessor): BrickNotTrainable(PreprocessorDummy, input_names=['raw'], output_names=['processed'], alive_stages=['TRAIN', 'VALIDATION', 'TEST', 'INFERENCE', 'EXPORT'])
#   (backbone): BrickTrainable(TinyModel, input_names=['processed'], output_names=['embedding'], alive_stages=['TRAIN', 'VALIDATION', 'TEST', 'INFERENCE', 'EXPORT'])
#   (head): BrickTrainable(ClassifierDummy, input_names=['embedding'], output_names=['logits', 'softmaxed'], alive_stages=['TRAIN', 'VALIDATION', 'TEST', 'INFERENCE', 'EXPORT'])
# )
```

All modules are added as entries in a regular dictionary and for each module, we wrap it inside a brick (here either `BrickTrainable` and 
`BrickNotTrainable`), provide a name (dictionary key) and input and output names. The number of input and output names should match the 
actually number of input and outputs for each function. 

Input and output names specify how tensors move between bricks: The `preprocessor` uses a `raw` input tensor and passes the
`processed` tensor to the `backbone`. The backbone returns the `embedding` tensor and passes it to the `head` determining 
both `logits` and `softmaxed` tensors. 

Bricks are passed to a `BrickCollection` for executing them as showed in below:

```python
batch_size=2
batch_images = torch.rand((batch_size, 3, 100, 200))
named_outputs = brick_collection(named_inputs={'raw': batch_images}, stage=Stage.INFERENCE)
print("Brick outputs:", named_outputs.keys())
# Brick outputs: dict_keys(['raw', 'stage', 'processed', 'embedding', 'logits', 'softmaxed'])
```

The brick collection accepts a dictionary and returns a dictionary with both intermediated and resulting tensors. 

Running our models as a brick collection has the following advantages:

- A brick collection act as a regular `nn.Module` with all the familiar features: a `forward`-function, a `to`-function to move 
  to a specific device/precision, you can save/load a model, management of parameters, onnx exportable etc. 
- A brick collection is also a simple DAG, it accepts a dictionary (`named_inputs`), 
executes each bricks and ensures that the outputs are passed to the inputs of other bricks with matching names. 
Structuring the model as a DAG, makes it easy to add/remove outputs for a given module during development, add new modules to the
collection and build completely new models from reusable parts. 
- A brick collection is actually a dictionary (`nn.DictModule`). Allowing you to access, pop and update the 
  collection easily as a regular dictionary. It can also handle nested dictionary, allowing groups of bricks to be added/removed easily. 

Note also that we set `stage=Stage.INFERENCE` to explicitly specify if we are doing training, validation, test or inference.
Specifying a stage is important, if we want a module to act in a specific way during specific stages.

Leading us to the next section


## Concept 2: Bricks can be dead or alive
The second concept is to specify when bricks are alive - meaning we specify at which stages (train, test, validation, inference and export) 
a brick is executed. For other stage the brick will play dead - do nothing / return empty dictionary. 

Meaning that for different `stages` of the model, we will have the option of creating a unique DAG for each model stage. 

In above example this is not particular interesting - because preprocessor, backbone model and head would typically be alive in all stages. 

So we will demonstrate by adding a loss brick (`BrickLoss`) and specifying `alive_stages` for each brick.

```python
num_classes = 3
bricks = {
    'preprocessor': BrickNotTrainable(PreprocessorDummy(), 
                                      input_names=['raw'], 
                                      output_names=['processed'], 
                                      alive_stages="all"),
    'backbone': BrickTrainable(TinyModel(n_channels=num_classes, n_features=10), 
                               input_names=['processed'], 
                               output_names=['embedding'], 
                               alive_stages="all"),
    'head': BrickTrainable(ClassifierDummy(num_classes=num_classes, in_features=10), 
                           input_names=['embedding'], 
                           output_names=['logits', 'softmaxed'], 
                           alive_stages="all"),
    'loss': BrickLoss(model=nn.CrossEntropyLoss(), 
                      input_names=['logits', 'targets'], 
                      output_names=['loss_ce'], 
                      alive_stages=[Stage.TRAIN, Stage.VALIDATION, Stage.TEST])
}
brick_collection = BrickCollection(bricks)

print(brick_collection)
# BrickCollection(
#   (preprocessor): BrickNotTrainable(PreprocessorDummy, input_names=['raw'], output_names=['processed'], alive_stages=['TRAIN', 'VALIDATION', 'TEST', 'INFERENCE', 'EXPORT'])
#   (backbone): BrickTrainable(TinyModel, input_names=['processed'], output_names=['embedding'], alive_stages=['TRAIN', 'VALIDATION', 'TEST', 'INFERENCE', 'EXPORT'])
#   (head): BrickTrainable(ClassifierDummy, input_names=['embedding'], output_names=['logits', 'softmaxed'], alive_stages=['TRAIN', 'VALIDATION', 'TEST', 'INFERENCE', 'EXPORT'])
#   (loss): BrickLoss(CrossEntropyLoss, input_names=['logits', 'targets'], output_names=['loss_ce'], alive_stages=['TRAIN', 'VALIDATION', 'TEST'])
# )
```

We set `preprocessor`, `backbone` and `head` to be alive on all stages `alive_stages="all"` - this is the default behavior and
similar to before. 

For `loss` we set `alive_stages=[Stage.TRAIN, Stage.VALIDATION, Stage.TEST]` to calculate loss during train, validation and test
stages. 

Another advantages is that model have different input requirements for different stages.

For `Stage.INFERENCE` and `Stage.EXPROT` stages, the model only requires the `raw` tensor as input. 

```python
named_outputs_without_loss = brick_collection(named_inputs={'raw': batch_images}, 
                                              stage=Stage.INFERENCE)   
```

<!-- #region -->


For `Stage.TRAIN`, `Stage.VALIDATION` and `Stage.TEST` stages, the model requires both `raw` and `targets` input tensors.
<!-- #endregion -->

```python
targets = torch.ones((batch_size,3))
named_outputs_with_loss = brick_collection(named_inputs={'raw': batch_images, "targets": targets}, 
                                           stage=Stage.TRAIN)
```

## Bricks for model training
We are not creating a training framework, but to easily use the brick collection in your favorite training framework or custom 
training/validation/test loop, we need the final piece. We should be able to calculate and gather metrics across a whole dataset. 

We will extend our example from before by adding metric bricks and common reusable components from `torchbricks.bag_of_bricks`.

```python
import torchvision
from torchbricks.bag_of_bricks import ImageClassifier, Preprocessor, resnet_to_brick
from torchbricks.bricks import BrickMetricSingle
from torchmetrics.classification import MulticlassAccuracy

num_classes = 10
resnet = torchvision.models.resnet18(weights=None, num_classes=num_classes)
resnet_brick = resnet_to_brick(resnet=resnet,  input_name='normalized', output_name='features')
n_features = resnet_brick.model.n_backbone_features
bricks = {
    'preprocessor': BrickNotTrainable(Preprocessor(), 
                                      input_names=['raw'], 
                                      output_names=['normalized']),
    'backbone': resnet_brick,
    'head': BrickTrainable(ImageClassifier(num_classes=num_classes, n_features=n_features),
                           input_names=['features'], 
                           output_names=['logits', 'probabilities', 'class_prediction']),
    'accuracy': BrickMetricSingle(MulticlassAccuracy(num_classes=num_classes), 
                                  input_names=['class_prediction', 'targets']),
    'loss': BrickLoss(model=nn.CrossEntropyLoss(), 
                      input_names=['logits', 'targets'], 
                      output_names=['loss_ce'])
}

brick_collection = BrickCollection(bricks)
named_inputs = {"raw": batch_images, "targets": torch.ones((batch_size), dtype=torch.int64)}
named_outputs = brick_collection(named_inputs=named_inputs, stage=Stage.TRAIN)
named_outputs = brick_collection(named_inputs=named_inputs, stage=Stage.TRAIN)
named_outputs = brick_collection(named_inputs=named_inputs, stage=Stage.TRAIN)
named_outputs = brick_collection(named_inputs=named_inputs, stage=Stage.TRAIN)
metrics = brick_collection.summarize(stage=Stage.TRAIN, reset=True)
print(f"{metrics=}, {named_outputs.keys()=}")
```


On each `forward`-call, we calculate model outputs, losses and metrics for each batch. Metrics are aggregated internally in `BrickMetricSingle` 
and only returned with the `summarize`-call. We set `reset=True` to reset metric aggregation. 

For metrics, we rely on the [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/) library and passes either a single 
metric (`torchmetrics.Metric`) to `BrickMetricSingle` or a collection of metrics (`torchmetrics.MetricCollection`) to `BrickMetrics`.

For multiple metrics, use always `BrickMetrics` with `torchmetrics.MetricCollection` [doc](https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metriccollection). 
It has some intelligent mechanisms for sharing 
metrics stats. 

Note also that metrics are not passed to other bricks - they are only stored internally. To also pass metrics to other bricks
(and add computational cost) you can set `return_metrics=True` for `BrickMetrics` and `BrickMetricSingle`.


## Bricks motivation (to be continued)

The main motivation:

- Each brick can return what ever - they are not forced to only returning e.g. logits... If you want the model backbone embeddings
  you can do that to. 
- Avoid modules within modules within modules to created models that are combined. 
- Not flexible. It is possible to make the first encode/decode model... But adding a preprocessor, swapping out a backbone,
  adding additional heads or necks and sharing computations will typically not be easy. I ended up creating multiple modules that are
  called within other modules... All head/modules pass dictionaries between modules. 
- Typically not very reusable. 
- By packing model modules, metrics and loss-functions into a brick collection, we can more easily 
inject any desired brick collection into your custom trainer without doing modifications to the trainer.

Including metrics and losses with the model. 
- Model, metrics and losses are connected. If we want to add an additional head to a model - we should also add losses and metrics. 
- The typical distinction between `encode`  / `decoder` becomes to limited... Multiple decoders might share a `neck`.


## Brick features: 

Missing sections:

- [x] Export as ONNX
- [x] Acts as a nn.Module
- [ ] Acts as a dictionary - Nested brick collection
- [ ] Training with Pytorch lightning
- [ ] Pass all inputs as a dictionary `input_names='all'`
- [ ] Using stage inside module
- [ ] the `extract_losses` function
- [ ] Bag of bricks - reusable bricks modules
  - [ ] Note also in above example we use bag-of-bricks to import commonly used `nn.Module`s. This includes a `Preprocessor`, `ImageClassifier` and `resnet_to_brick` to convert torchvision resnet models into a backbone brick  (without a classifier).
- [ ] The default `BrickModule`
- [ ] In this example we do not use `BrickModule` to build our collection - you can do that -
but instead we recommend using our pre-configured brick modules (`BrickLoss`, `BrickNotTrainable`, `BrickTrainable`, 
`BrickMetricSingle` and `BrickCollection`) to both ensure sensible defaults and to show the intend of each brick. 


### Brick features: Export as ONNX
To export a brick collection as onnx we provide the `export_bricks_as_onnx`-function. 

Pass an example input (`named_input`) to trace a brick collection.
Set `dynamic_batch_size=True` to support any batch size inputs and here we explicitly set `stage=Stage.EXPORT` - this is also 
the default.

```python
from pathlib import Path
from torchbricks.brick_utils import export_bricks_as_onnx
path_onnx = Path("build/readme_model.onnx")
export_bricks_as_onnx(path_onnx=path_onnx, 
                      brick_collection=brick_collection, 
                      named_inputs=named_inputs, 
                      dynamic_batch_size=True, 
                      stage=Stage.EXPORT)
```

### Brick features: Act as a nn.Module
A brick collection acts as a 'nn.Module' meaning:

```python
# Move to specify device (CPU/GPU) or precision to automatically move model parameters
brick_collection.to(torch.float16)
brick_collection.to(torch.float32)

# Save model parameters
path_model = Path("build/readme_model.pt")
torch.save(brick_collection.state_dict(), path_model)

# Load model parameters
brick_collection.load_state_dict(torch.load(path_model))

# Access parameters
brick_collection.named_parameters()

# Using compile with pytorch >= 2.0
torch.compile(brick_collection)
```

### Brick features: Act as a dictionary (nn.ModuleDict) and relative input/output names



```python
from typing import Dict

from torchbricks.bricks import BrickInterface


def image_classifier_head(num_classes: int, in_channels: int, input_name: str) -> Dict[str, BrickInterface]:
    """Image classifier bricks: Classifier, loss and metrics """
    head = {
        'classify': BrickTrainable(ImageClassifier(num_classes=num_classes, n_features=in_channels),
                                   input_names=[input_name], 
                                   output_names=['./logits', './probabilities', './class_prediction']),
        'accuracy': BrickMetricSingle(MulticlassAccuracy(num_classes=num_classes), 
                                      input_names=['./class_prediction', 'targets']),
        'loss': BrickLoss(model=nn.CrossEntropyLoss(),
                          input_names=['./logits', 'targets'], 
                          output_names=['./loss_ce'])
    }
    return head

n_features = resnet_brick.model.n_backbone_features
bricks = {
    'preprocessor': BrickNotTrainable(Preprocessor(), 
                                      input_names=['raw'], 
                                      output_names=['normalized']),
    'backbone': resnet_brick,
    'head0': image_classifier_head(num_classes=3, in_channels=n_features, input_name='features'),
    'head1': image_classifier_head(num_classes=5, in_channels=n_features, input_name='features'),
}
print(BrickCollection(bricks))
# BrickCollection(
#   (preprocessor): BrickNotTrainable(Preprocessor, input_names=['raw'], output_names=['normalized'], alive_stages=['TRAIN', 'VALIDATION', 'TEST', 'INFERENCE', 'EXPORT'])
#   (backbone): BrickTrainable(BackboneResnet, input_names=['normalized'], output_names=['features'], alive_stages=['TRAIN', 'VALIDATION', 'TEST', 'INFERENCE', 'EXPORT'])
#   (head0): BrickCollection(
#     (classify): BrickTrainable(ImageClassifier, input_names=['features'], output_names=['head0/logits', 'head0/probabilities', 'head0/class_prediction'], alive_stages=['TRAIN', 'VALIDATION', 'TEST', 'INFERENCE', 'EXPORT'])
#     (accuracy): BrickMetricSingle(['MulticlassAccuracy'], input_names=['head0/class_prediction', 'targets'], output_names=[], alive_stages=['TRAIN', 'TEST', 'VALIDATION'])
#     (loss): BrickLoss(CrossEntropyLoss, input_names=['head0/logits', 'targets'], output_names=['head0/loss_ce'], alive_stages=['TRAIN', 'TEST', 'VALIDATION'])
#   )
#   (head1): BrickCollection(
#     (classify): BrickTrainable(ImageClassifier, input_names=['features'], output_names=['head1/logits', 'head1/probabilities', 'head1/class_prediction'], alive_stages=['TRAIN', 'VALIDATION', 'TEST', 'INFERENCE', 'EXPORT'])
#     (accuracy): BrickMetricSingle(['MulticlassAccuracy'], input_names=['head1/class_prediction', 'targets'], output_names=[], alive_stages=['TRAIN', 'TEST', 'VALIDATION'])
#     (loss): BrickLoss(CrossEntropyLoss, input_names=['head1/logits', 'targets'], output_names=['head1/loss_ce'], alive_stages=['TRAIN', 'TEST', 'VALIDATION'])
#   )
# )
```

### Bag of bricks - reusable bricks modules
Note also in above example we use bag-of-bricks to import commonly used `nn.Module`s 

This includes a `Preprocessor`, `ImageClassifier` and `resnet_to_brick` to convert torchvision resnet models into a backbone brick 
(without a classifier).


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

trainer = pl.Trainer(max_epochs=1, limit_train_batches=2, limit_val_batches=2, limit_test_batches=2)
# To train and test model
trainer.fit(bricks_lightning_module, datamodule=data_module)
trainer.test(bricks_lightning_module, datamodule=data_module)
```




Is mermaid rendered in GITHUB? 
```mermaid
flowchart LR
    c1-->a2
    subgraph one
    a1-->a2
    end
    subgraph two
    b1-->b2
    end
    subgraph three
    c1-->c2
    end
    one --> two
    three --> two
    two --> c2
``````



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
- [x] Add typeguard
- [x] Allow a brick to receive all named_inputs and add a test for it.
- [x] Fix the release process. It should be as simple as running `make release`.
- [x] Add onnx export example to the README.md
- [x] Pretty print bricks
- [x] Relative input/output names
- [x] Test to verify that environment matches conda lock. The make command 'update-lock-file' should store a copy of 'environment.yml'
      We will the have a test checking if the copy and the current version of `environment.yml` is the same.
- [ ] Create brick-collection visualization tool ("mermaid?")
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
  - [ ] Maybe visualizations should be done in OpenCV it is faster. 
- [ ] Support torch.jit.scripting? 

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


