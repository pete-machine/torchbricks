import pytest
import torch
import torchvision
from torchbricks.bag_of_bricks.backbones import SUPPORTED_BACKBONES, create_backbone, resnet_to_brick
from torchbricks.bricks import Stage


def test_convert_resnet_backbone_brick():
    resnet_model = torchvision.models.resnet18(weights=None, num_classes=10)
    resnet_backbone_brick = resnet_to_brick(resnet=resnet_model, input_name="image", output_name="features")

    output = resnet_backbone_brick(named_inputs={"image": torch.rand((1, 3, 50, 100))}, stage=Stage.TRAIN)
    assert hasattr(resnet_backbone_brick.model, "n_backbone_features")
    assert resnet_backbone_brick.model.n_backbone_features == output["features"].shape[1]


@pytest.mark.parametrize("backbone_name", SUPPORTED_BACKBONES)
def test_create_backbone(backbone_name):
    backbone = create_backbone(name=backbone_name, pretrained=False)

    output = backbone(torch.rand((1, 3, 50, 100)))
    assert hasattr(backbone, "n_backbone_features")
    assert backbone.n_backbone_features == output.shape[1]
