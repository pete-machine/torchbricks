import pytest
import torch
import torchvision
from torchbricks.bag_of_bricks import SUPPORTED_BACKBONES, ImageClassifier, create_backbone, resnet_to_brick
from torchbricks.bricks import Stage


def test_convert_resnet_backbone_brick():

    resnet_model = torchvision.models.resnet18(weights=None, num_classes=10)
    resnet_backbone_brick = resnet_to_brick(resnet=resnet_model, input_name='image', output_name='features')

    output = resnet_backbone_brick(named_inputs={'image': torch.rand((1, 3, 50, 100))}, stage=Stage.TRAIN)
    assert hasattr(resnet_backbone_brick.model, 'n_backbone_features')
    assert resnet_backbone_brick.model.n_backbone_features == output['features'].shape[1]

@pytest.mark.parametrize('backbone_name', SUPPORTED_BACKBONES)
def test_create_backbone(backbone_name):
    backbone = create_backbone(name=backbone_name, pretrained=True)

    output = backbone(torch.rand((1, 3, 50, 100)))
    assert hasattr(backbone, 'n_backbone_features')
    assert backbone.n_backbone_features == output.shape[1]



def test_image_classifier_average_pooling():
    batch_size = 2
    input_features = 5
    n_classes = 10
    HW=20
    image_classifier = ImageClassifier(num_classes=n_classes, n_features=input_features, use_average_pooling=True)
    embedding = torch.zeros((batch_size, input_features, HW, HW))
    logits, probabilities, class_prediction = image_classifier(embedding)

    assert logits.shape == (batch_size, n_classes)
    assert probabilities.shape == (batch_size, n_classes)
    assert class_prediction.shape == (batch_size, )

def test_image_classifier_no_average_pooling():
    batch_size = 2
    input_features = 5
    n_classes = 10
    image_classifier = ImageClassifier(num_classes=n_classes, n_features=input_features, use_average_pooling=False)
    embedding = torch.zeros((batch_size, input_features))
    logits, probabilities, class_prediction = image_classifier(embedding)

    assert logits.shape == (batch_size, n_classes)
    assert probabilities.shape == (batch_size, n_classes)
    assert class_prediction.shape == (batch_size, )
