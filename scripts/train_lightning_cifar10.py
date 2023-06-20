import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningDataModule, Trainer
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
import torchmetrics
from torchmetrics import classification
import torchvision

from torchvision import datasets, transforms
from lightning_module import LightningBrickCollection
from torchbricks.bag_of_bricks import resnet_to_brick
from torchbricks.bag_of_bricks import ImageClassifier

from torchbricks.bricks import Brick, BrickCollection, BrickLoss, BrickNotTrainable, BrickTorchMetric, BrickTrainable, Phase
from torchbricks.custom_metrics import ConcatenatePredictionAndTarget


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


class PreprocessorCifar10(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

    def forward(self, x):
        return self.normalize(x)


def create_resnet_18(pretrained=False, num_classes=10):
    model = torchvision.models.resnet18(pretrained=pretrained, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


def create_cifar_bricks(num_classes: int) -> Dict[str, Brick]:

    named_bricks = {
        'preprocessor': BrickNotTrainable(PreprocessorCifar10(), input_names=['raw'], output_names=['normalized']),
        'backbone': resnet_to_brick(resnet=create_resnet_18(num_classes=num_classes),
                                                     input_name='normalized',
                                                     output_name='features'),
    }

    n_backbone_features = named_bricks['backbone'].model.n_backbone_features

    metrics = torchmetrics.MetricCollection({
        'MeanAccuracy': classification.MulticlassAccuracy(num_classes=num_classes, average='macro', multiclass=True),
        'Accuracy': classification.MulticlassAccuracy(num_classes=num_classes, average='micro', multiclass=True),
        'ConfMat': torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes),
        'Concatenate': ConcatenatePredictionAndTarget(compute_on_cpu=True)
    })
    named_bricks['task_classifier'] = {
        'classifier': BrickTrainable(ImageClassifier(num_classes=num_classes, n_features=n_backbone_features),
                                     input_names=['features'], output_names=['logits', 'probabilities', 'class_prediction']),
        'loss_ce': BrickLoss(model=nn.CrossEntropyLoss(), input_names=['logits', 'targets'], output_names=['loss_ce']),
        'metrics_classification': BrickTorchMetric(metric=metrics, input_names=['class_prediction', 'targets'],  metric_name=''),
        }


    return named_bricks


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


def create_optimizers(model_parameters,
                      max_epochs: int,
                      n_steps_per_epoch: int,
                      lr: float = 0.05,
                      momentum: float = 0.9,
                      weight_decay: float = 5e-4):
    optimizer = torch.optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = create_lr_schedular_one_cycle_lr(optimizer=optimizer, max_epochs=max_epochs, steps_per_epoch=n_steps_per_epoch)
    return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            }


if __name__ == '__main__':
    PROJECT = 'CIFAR10'
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=40)
    parser.add_argument('--accelerator', type=str, default='gpu')
    args = parser.parse_args()

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
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    test_transforms=test_transforms,
                                    train_transforms=train_transforms)
    data_module.prepare_data()
    data_module.setup()
    n_channels, height, width = data_module.dims
    num_classes = int(data_module.num_classes)

    train_dataloader = data_module.train_dataloader()
    n_steps_per_epoch = len(train_dataloader)
    for batch in train_dataloader:
        named_inputs = {'raw': batch[0], 'targets': batch[1]}
        break
    brick_collection = BrickCollection(create_cifar_bricks(num_classes=num_classes))
    named_outputs, loss = brick_collection.on_step(phase=Phase.TRAIN, named_inputs=named_inputs, batch_idx=0)
    named_outputs, loss = brick_collection.on_step(phase=Phase.TRAIN, named_inputs=named_inputs, batch_idx=0)
    named_outputs, loss = brick_collection.on_step(phase=Phase.TRAIN, named_inputs=named_inputs, batch_idx=0)
    metrics_train = brick_collection.summarize(phase=Phase.TRAIN, reset=True)

    # Forward only runs inference stuff. No metrics, no losses and does not require targets.
    named_inputs_no_target = {'raw': batch[0]}
    brick_collection(phase=Phase.TEST, named_inputs=named_inputs_no_target)

    def create_optimizers_func(params):
        return create_optimizers(model_parameters=params, max_epochs=args.num_workers, n_steps_per_epoch=n_steps_per_epoch)

    path_experiments = Path('runs')
    bricks_lightning_module = LightningBrickCollection(path_experiments=path_experiments,
                                                       experiment_name=experiment_name,
                                                       brick_collection=brick_collection,
                                                       create_optimizers_func=create_optimizers_func)
    logger = WandbLogger(name=experiment_name, project=PROJECT)
    trainer = Trainer(accelerator=args.accelerator, logger=logger, max_epochs=args.max_epochs)
    trainer.fit(bricks_lightning_module,
                train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader())
    trainer.test(bricks_lightning_module, datamodule=data_module)
