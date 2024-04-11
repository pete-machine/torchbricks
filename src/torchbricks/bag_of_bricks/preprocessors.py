from torch import nn
from torchvision import transforms


class Preprocessor(nn.Module):
    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        uint8_to_float32: bool = False,
    ):
        super().__init__()
        self.transforms = transforms.Normalize(mean=mean, std=std)
        self.uint8_to_float32 = uint8_to_float32

    def __call__(self, img):
        if self.uint8_to_float32:
            img = img.float() / 255.0
        return self.transforms(img)
