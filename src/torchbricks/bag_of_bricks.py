import torch
from torch import Tensor, nn
from torchvision import transforms
from torchvision.models import ResNet

from torchbricks.bricks import BrickTrainable


def resnet_to_brick(resnet: ResNet, input_name: str, output_name: str):
    """Function to convert a torchvision resnet model and convert it to a torch model"""
    return BrickTrainable(model=BackboneResnet(resnet), input_names=[input_name], output_names=[output_name])

class BackboneResnet(nn.Module):
    def __init__(self, resnet: ResNet) -> None:
        super().__init__()
        self.resnet = resnet
        self.n_backbone_features = list(resnet.layer4.children())[-1].conv1.weight.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        # Similar to 'torchvision/models/resnet.py' but without the final average pooling and classification layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

class ImageClassifier(nn.Module):
    """"""
    def __init__(self, num_classes: int, n_features: int, use_average_pooling: bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(n_features, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.use_average_pooling = use_average_pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, backbone_features):
        if self.use_average_pooling:
            backbone_features = self.avgpool(backbone_features)
        x = torch.flatten(backbone_features, 1)
        logits = self.fc(x)
        probabilities = self.softmax(logits)
        class_prediction = torch.argmax(probabilities, dim=1)
        return logits, probabilities, class_prediction


class Preprocessor(nn.Module):
    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.transforms = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img):
        return self.transforms(img)
