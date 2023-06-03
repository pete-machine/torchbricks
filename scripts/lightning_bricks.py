import argparse
from datetime import datetime
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import torch
import torchvision
from functools import partial
from lightning_module import CIFAR10DataModule, LightningBrickCollection, create_cifar_bricks, create_lr_schedular_one_cycle_lr

from torch_bricks.bricks import BrickCollection, Phase



if __name__ == '__main__':
    PROJECT = 'CIFAR10'
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=10)
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
        named_tensors = {'raw': batch[0], 'targets': batch[1]}
        break
    brick_collection = BrickCollection(create_cifar_bricks(num_classes=num_classes))
    named_tensors, loss = brick_collection.on_step(phase=Phase.TRAIN, named_tensors=named_tensors, batch_idx=0)
    named_tensors, loss = brick_collection.on_step(phase=Phase.TRAIN, named_tensors=named_tensors, batch_idx=0)
    named_tensors, loss = brick_collection.on_step(phase=Phase.TRAIN, named_tensors=named_tensors, batch_idx=0)
    metrics_train = brick_collection.summarize(phase=Phase.TRAIN, reset=True)

    # Forward only runs inference stuff. No metrics, no losses and does not require targets.
    named_tensors_no_target = named_tensors = {'raw': batch[0]}
    brick_collection(phase=Phase.TEST, named_tensors=named_tensors_no_target)

    create_opimtizer_func = partial(torch.optim.SGD, lr=0.05, momentum=0.9, weight_decay=5e-4)
    create_lr_scheduler = partial(create_lr_schedular_one_cycle_lr, max_epochs=args.max_epochs, steps_per_epoch=n_steps_per_epoch)
    path_experiments = Path('runs')
    bricks_lightning_module = LightningBrickCollection(path_experiments=path_experiments,
                                                       experiment_name=experiment_name,
                                                       brick_collection=brick_collection,
                                                       create_optimizer_func=create_opimtizer_func,
                                                       create_lr_scheduler_func=create_lr_scheduler)
    logger = WandbLogger(name=experiment_name, project=PROJECT)
    trainer = Trainer(accelerator=args.accelerator, logger=logger, max_epochs=args.max_epochs)
    # trainer.logger.log_hyperparams(cfg_log)
    trainer.fit(bricks_lightning_module,
                train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader())
    trainer.test(bricks_lightning_module, datamodule=data_module)
