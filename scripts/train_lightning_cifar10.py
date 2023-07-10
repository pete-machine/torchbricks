import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torchmetrics
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torchbricks.bag_of_bricks import ImageClassifier, Preprocessor, resnet_to_brick
from torchbricks.bricks import BrickCollection, BrickInterface, BrickLoss, BrickMetricMultiple, BrickNotTrainable, BrickTrainable
from torchbricks.custom_metrics import ConcatenatePredictionAndTarget
from torchmetrics import classification
from utils_testing.datamodule_cifar10 import CIFAR10DataModule
from utils_testing.lightning_module import LightningBrickCollection


def create_resnet_18(weights=None, num_classes=10):
    model = torchvision.models.resnet18(weights=weights, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


def create_cifar_bricks(num_classes: int) -> Dict[str, BrickInterface]:

    named_bricks = {
        'preprocessor': BrickNotTrainable(Preprocessor(), input_names=['raw'], output_names=['normalized']),
        'backbone': resnet_to_brick(resnet=create_resnet_18(num_classes=num_classes), input_name='normalized', output_name='features'),
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
        'metrics_classification': BrickMetricMultiple(metric_collection=metrics, input_names=['class_prediction', 'targets']),
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

    brick_collection = BrickCollection(create_cifar_bricks(num_classes=num_classes))

    def create_optimizers_func(params):
        return create_optimizers(model_parameters=params, max_epochs=args.max_epochs, n_steps_per_epoch=n_steps_per_epoch)

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
