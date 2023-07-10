from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Note: Use 'from pl_bolts.datamodules import CIFAR10DataModule' in the future. Currently, it is not working with my current package

class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str,
                 batch_size: int,
                 num_workers: int,
                 train_transforms: transforms.Compose,
                 test_transforms: transforms.Compose,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.label_names = [str(number) for number in range(self.num_classes)]

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
