import os
from pathlib import Path
from typing import Callable, Dict
from pytorch_lightning import LightningModule, LightningDataModule
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
import torchmetrics
from torchmetrics import classification
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

from torchbricks.bricks import Brick, BrickCollection, BrickLoss, BrickNotTrainable, BrickTorchMetric, BrickTrainable, Phase
from torchbricks.custom_metrics import ConcatenatePredictionAndTarget



def create_lr_schedular_one_cycle_lr(optimizer, max_epochs: int, steps_per_epoch: int):
    scheduler_dict = {
        'scheduler': OneCycleLR(
            optimizer,
            0.1,
            epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
        ),
        'interval': 'step',
    }
    return scheduler_dict


class LightningBrickCollection(LightningModule):
    def __init__(self, path_experiments: Path,
                 experiment_name: str,
                 brick_collection: BrickCollection,
                 create_optimizer_func: Callable,
                 create_lr_scheduler_func: Callable = None):
        super().__init__()

        self.path_experiment = path_experiments / experiment_name
        os.mkdir(self.path_experiment)
        self.epoch_count = 0

        self.bricks = brick_collection
        self.create_optimizer_func = create_optimizer_func
        self.create_lr_scheduler_func = create_lr_scheduler_func


    def _on_epoch_end(self, phase: Phase):
        metrics = self.bricks.summarize(phase=phase, reset=True)
        def is_single_value_tensor(tensor):
            return isinstance(tensor, torch.Tensor) and tensor.ndim == 0

        single_valued_metrics = {metric_name: value for metric_name, value in metrics.items() if is_single_value_tensor(value)}
        self.log_dict(single_valued_metrics)

    def _step(self, phase: Phase, batch, batch_idx: int):
        named_inputs = {'raw': batch[0], 'targets': batch[1]}
        _, losses = self.bricks.on_step(phase=phase, named_inputs=named_inputs, batch_idx=batch_idx)
        loss = 0
        for loss_name, loss_value in losses.items():
            self.log(f'{phase.value}/{loss_name}', loss_value)
            loss = loss + loss_value
        self.log(f'{phase.value}/total_loss', loss)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self._step(phase=Phase.TRAIN, batch=batch, batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx: int):
        self._step(phase=Phase.VALIDATION, batch=batch, batch_idx=batch_idx)

    def test_step(self, batch, batch_idx):
        self._step(phase=Phase.TEST, batch=batch, batch_idx=batch_idx)

    def on_train_epoch_end(self):
        self._on_epoch_end(phase=Phase.TRAIN)
        self.epoch_count = self.epoch_count+1

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(phase=Phase.VALIDATION)

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(phase=Phase.TEST)

    def configure_optimizers(self):
        optimizer = self.create_optimizer_func(self.parameters())
        optimizers = {
            'optimizer': optimizer,
            'lr_scheduler': self.create_lr_scheduler_func(optimizer)
            }
        return optimizers

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



def create_cifar_bricks(num_classes: int) -> Dict[str, Brick]:
    backbone = create_resnet_18()
    metric_collection = torchmetrics.MetricCollection({
        'MeanAccuracy': classification.MulticlassAccuracy(num_classes=num_classes, average='macro', multiclass=True),
        'Accuracy': classification.MulticlassAccuracy(num_classes=num_classes, average='micro', multiclass=True),
        'ConfMat': torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes),
        'Concatenate': ConcatenatePredictionAndTarget(compute_on_cpu=True)
    })
    named_bricks = {
        'preprocessor': BrickNotTrainable(PreprocessorCifar10(), input_names=['raw'], output_names=['normalized']),
        'backbone': BrickTrainable(backbone, input_names=['normalized'], output_names=['backbone']),
        'classifier': BrickTrainable(Classifier(num_classes=num_classes, n_features=backbone.n_backbone_features),
                                     input_names=['backbone'], output_names=['logits', 'probabilities', 'class_prediction']),
        'loss_ce': BrickLoss(model=nn.CrossEntropyLoss(), input_names=['logits', 'targets'], output_names=['loss_ce']),
        'metrics_classification': BrickTorchMetric(metric=metric_collection,
                                                   input_names=['class_prediction', 'targets'],  metric_name=''),
    }
    return named_bricks


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
