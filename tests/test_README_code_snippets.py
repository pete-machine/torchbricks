
import torch
from torch import nn

class Preprocessor(nn.Module):
    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        return raw_input/2

class TinyModel(nn.Module):
    def __init__(self, n_channels: int, n_features: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_features, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv(tensor)

class Classifier(nn.Module):
    def __init__(self, num_classes: int, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.avgpool(tensor)))

def test_basic_use_case_image_classification():
    from torchbricks.bricks import BrickCollection, BrickNotTrainable, BrickTrainable, Phase

    # Defining model from bricks
    bricks = {
        'preprocessor': BrickNotTrainable(Preprocessor(),
                                          input_names=['raw'], output_names=['processed']),
        'backbone': BrickTrainable(TinyModel(n_channels=3, n_features=10),
                                   input_names=['processed'], output_names=['embedding']),
        'classifier': BrickTrainable(Classifier(num_classes=3, in_features=10),
                                     input_names=['embedding'], output_names=['logits'])
    }

    # Executing model
    model = BrickCollection(bricks)
    batch_image_example = torch.rand((1, 3, 100, 200))
    outputs = model(named_inputs={'raw': batch_image_example}, phase=Phase.TRAIN)
    print(outputs.keys())
