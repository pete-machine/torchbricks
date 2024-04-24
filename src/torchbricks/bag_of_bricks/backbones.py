import torchvision
from torch import Tensor, nn
from torchbricks.bricks import BrickTrainable
from torchvision.models import ResNet

SUPPORTED_RESNET_BACKBONES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]
SUPPORTED_BACKBONES = SUPPORTED_RESNET_BACKBONES


class BackboneResnet(nn.Module):
    def __init__(self, resnet: ResNet) -> None:
        super().__init__()
        delattr(resnet, "fc")  # The final classification layer is not used. Remove or unused parameters will raise model warnings/errors
        delattr(resnet, "avgpool")  # Remove the final average pooling layer. It is not used either

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


def create_backbone(name: str, pretrained: bool, small: bool = False) -> nn.Module:
    """Creates a backbone from a name without classification head"""
    weights = "DEFAULT" if pretrained else None
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Backbone {name} not supported. Supported backbones are {SUPPORTED_BACKBONES}")

    if name in SUPPORTED_RESNET_BACKBONES:
        model = torchvision.models.resnet.__dict__[name](weights=weights)
        if small:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        return BackboneResnet(resnet=model)


def resnet_to_brick(resnet: ResNet, input_name: str, output_name: str):
    """Function to convert a torchvision resnet model and convert it to a torch model"""
    return BrickTrainable(model=BackboneResnet(resnet), input_names=[input_name], output_names=[output_name])
