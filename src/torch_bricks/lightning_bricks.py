from datetime import datetime
import os
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
import torchmetrics
from torchmetrics import classification
import torchvision
from torchvision import datasets, transforms

from torch.utils.data import random_split, DataLoader

from bricks import BrickCollection, BrickLoss, BrickNotTrainable, BrickTorchMetric, BrickTrainable, RunState

import utils
# from pl_bolts.datamodules import CIFAR10DataModule THIS NEEDS A FIX APPARENTLY
# from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

# @typechecked


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str,
                 batch_size: int,
                 num_workers: int,
                 train_transforms: transforms.Compose,
                 test_transforms: transforms.Compose,
                 # val_transforms: transforms.Compose,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        # self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.label_names = list(str(number) for number in range(self.num_classes))

    def get_data_class_info(self):
        return {'dims': self.dims, 'num_classes': self.num_classes, 'label_names': self.label_names}

    def prepare_data(self):
        # download
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_full = datasets.CIFAR10(self.data_dir, train=True, transform=self.train_transforms)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar_test = datasets.CIFAR10(self.data_dir, train=False, transform=self.test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


def create_resnet_18(pretrained=False, num_classes=10):
    model = torchvision.models.resnet18(pretrained=pretrained, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Identity()
    model.n_backbone_features = list(model.layer4.children())[-1].conv1.weight.shape[1]
    return model


class Classifier(nn.Module):
    def __init__(self, num_classes: int, n_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(n_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, backbone_features):
        logits = self.fc(backbone_features)
        probabilities = self.softmax(logits)
        class_prediction = torch.argmax(probabilities, dim=1)
        return logits, probabilities, class_prediction


class PreprocessorCifar10(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

    def forward(self, x):
        return self.normalize(x)


class LightningBrickCollection(LightningModule):
    def __init__(self, path_experiments: Path, experiment_name: str, brick_collection: BrickCollection):
        super().__init__()

        self.path_experiment = path_experiments / experiment_name
        os.mkdir(self.path_experiment)
        self.epoch_count = 0

        self.bricks = brick_collection
        # self.cfg_optimizer = cfg_optimizer


    def _on_epoch_end(self, state: RunState):
        metrics = self.bricks.summarize(state=state, reset=True)
        def is_single_value_tensor(tensor):
            return isinstance(tensor, torch.Tensor) and tensor.ndim == 0

        single_valued_metrics = {metric_name: value for metric_name, value in metrics.items() if is_single_value_tensor(value)}
        self.log_dict(single_valued_metrics)

    def _step(self, state: RunState, batch, batch_idx: int):
        named_tensors = {'raw': batch[0], 'targets': batch[1]}
        _, losses = self.bricks.on_step(state=state, named_tensors=named_tensors, batch_idx=batch_idx)
        loss = 0
        for loss_name, loss_value in losses.items():
            self.log(f'{state.value}/{loss_name}', loss_value)
            loss = loss + loss_value
        self.log(f'{state.value}/total_loss', loss)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self._step(state=RunState.TRAIN, batch=batch, batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx: int):
        self._step(state=RunState.VALIDATION, batch=batch, batch_idx=batch_idx)

    def test_step(self, batch, batch_idx):
        self._step(state=RunState.TEST, batch=batch, batch_idx=batch_idx)

    def on_train_epoch_end(self):
        self._on_epoch_end(state=RunState.TRAIN)
        self.epoch_count = self.epoch_count+1

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(state=RunState.VALIDATION)

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(state=RunState.TEST)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.05,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            'scheduler': OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

# class UnpackBatch(nn.Module):
#     def forward(self, state: RunState, batch: Tuple) -> torch.Tensor:

#         if state == RunState.INFERENCE:
#             return batch,
#         batch[0]
#         return {""}

if __name__ == '__main__':

    BATCH_SIZE=256  # 256
    NUM_WORKERS=10  # 0
    PROJECT='CIFAR10'
    ACCELERATOR='gpu'
    MAX_EPOCHS=40  # 20

    experiment_name = datetime.now().strftime('%Y_%m_%d_T_%H_%M_%S')
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    data_module = CIFAR10DataModule(data_dir='data',
                                    batch_size=BATCH_SIZE,
                                    num_workers=NUM_WORKERS,
                                    test_transforms=test_transforms,
                                    train_transforms=train_transforms)
    data_module.prepare_data()
    data_module.setup()
    n_channels, height, width = data_module.dims
    num_classes = int(data_module.num_classes)
    backbone = create_resnet_18()
    metric_collection = torchmetrics.MetricCollection({
        'MeanAccuracy': classification.MulticlassAccuracy(num_classes=num_classes, average='macro', multiclass=True),
        'Accuracy': classification.MulticlassAccuracy(num_classes=num_classes, average='micro', multiclass=True),
        'ConfMat': torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes),
        'Concatenate': utils.ConcatenatePredictionAndTarget(compute_on_cpu=True)
    })
    task_cls = BrickCollection({
        'classifier': BrickTrainable(Classifier(num_classes=num_classes, n_features=backbone.n_backbone_features),
                                     input_names=['backbone'], output_names=['logits', 'probabilities', 'class_prediction']),
        'loss_bce': BrickLoss(model=nn.CrossEntropyLoss(), input_names=['logits', 'targets'], output_names=['loss_bce']),
        'metrics_classification': BrickTorchMetric(metric=metric_collection,
                                                   input_names=['class_prediction', 'targets'],  metric_name=''),
        # 'metrics_loss': BrickTorchMetric(metric=torchmetrics.MeanMetric(), input_names=['loss_bce'], metric_name='loss'),

    })

    named_bricks = {
        'preprocessor': BrickNotTrainable(PreprocessorCifar10(), input_names=['raw'], output_names=['normalized']),
        'backbone': BrickTrainable(backbone, input_names=['normalized'], output_names=['backbone']),
        # 'head_classifier': task_cls,
        'classifier': BrickTrainable(Classifier(num_classes=num_classes, n_features=backbone.n_backbone_features),
                                     input_names=['backbone'], output_names=['logits', 'probabilities', 'class_prediction']),
        'loss_bce': BrickLoss(model=nn.CrossEntropyLoss(), input_names=['logits', 'targets'], output_names=['loss_bce']),
        'metrics_classification': BrickTorchMetric(metric=metric_collection,
                                                   input_names=['class_prediction', 'targets'],  metric_name=''),
        # 'metrics_loss': BrickTorchMetric(metric=torchmetrics.MeanMetric(), input_names=['loss_bce'], metric_name='loss'),
    }

    train_dataloader = data_module.train_dataloader()
    for batch in train_dataloader:
        named_tensors = {'raw': batch[0], 'targets': batch[1]}
        break
    brick_collection = BrickCollection(named_bricks)
    named_tensors, loss = brick_collection.on_step(state=RunState.TRAIN, named_tensors=named_tensors, batch_idx=0)
    named_tensors, loss = brick_collection.on_step(state=RunState.TRAIN, named_tensors=named_tensors, batch_idx=0)
    named_tensors, loss = brick_collection.on_step(state=RunState.TRAIN, named_tensors=named_tensors, batch_idx=0)
    metrics_train = brick_collection.summarize(state=RunState.TRAIN, reset=True)
    # metrics_test = brick_collection.summaries(state=RunState.TEST, reset=True) -> Error

    # Forward only runs inference stuff. No metrics, no losses and does not require targets.
    named_tensors_no_target = named_tensors = {'raw': batch[0]}
    brick_collection(state=RunState.TEST, named_tensors=named_tensors_no_target)


    path_experiments = Path('runs')
    bricks_lightning_module = LightningBrickCollection(path_experiments=path_experiments,
                                                       experiment_name=experiment_name,
                                                       brick_collection=brick_collection)
    logger = WandbLogger(name=experiment_name, project=PROJECT)
    trainer = Trainer(accelerator=ACCELERATOR, logger=logger, max_epochs=MAX_EPOCHS)
    # trainer.logger.log_hyperparams(cfg_log)
    trainer.fit(bricks_lightning_module,
                train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader())
    trainer.test(bricks_lightning_module, datamodule=data_module)
