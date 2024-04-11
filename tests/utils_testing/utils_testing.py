from pathlib import Path
from typing import Dict

import torch
import torchmetrics
from torch import nn
from torchbricks import custom_metrics
from torchbricks.bricks import BrickInterface, BrickLoss, BrickMetrics, BrickNotTrainable, BrickTrainable
from torchmetrics.classification import MulticlassAccuracy


def path_repo_root():
    return Path(__file__).parents[2]


def assert_equal_dictionaries(d0: Dict, d1: Dict, is_close: bool = False):
    assert set(d0) == set(d1)
    for key, values in d0.items():
        if isinstance(values, torch.Tensor):
            if is_close:
                assert torch.allclose(values, d1[key])
            else:
                assert values.equal(d1[key])
        else:
            assert values == d1[key]


def create_dummy_brick_collection(num_classes: int, num_backbone_featues: int) -> Dict[str, BrickInterface]:
    class Preprocessor(nn.Module):
        def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
            return raw_input / 2

    class TinyBackbone(nn.Module):
        def __init__(self, n_kernels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_kernels, kernel_size=3, padding=1)

        def forward(self, tensor: torch.Tensor):
            return self.conv1(tensor)

    class Classifier(nn.Module):
        def __init__(self, input_channels: int, num_classes: int):
            super().__init__()
            self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Conv2d(in_channels=input_channels, out_channels=num_classes, kernel_size=1)

        def forward(self, features: torch.Tensor):
            logits = torch.squeeze(self.classifier(self.global_average_pooling(features)))
            return logits

    metric_collection = torchmetrics.MetricCollection(
        {
            "MeanAccuracy": MulticlassAccuracy(num_classes=num_classes, average="macro"),
            "Accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
            "ConfMat": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes),
            "Concatenate": custom_metrics.ConcatenatePredictionAndTarget(compute_on_cpu=True),
        }
    )
    classifier = Classifier(input_channels=num_backbone_featues, num_classes=num_classes)

    bricks = {
        "preprocessor": BrickNotTrainable(Preprocessor(), input_names=["raw"], output_names=["preprocessed"]),
        "backbone": BrickTrainable(TinyBackbone(n_kernels=num_backbone_featues), input_names=["preprocessed"], output_names=["features"]),
    }
    bricks["head"] = {
        "classifier": BrickTrainable(classifier, input_names=["features"], output_names=["predictions"]),
        "loss": BrickLoss(nn.CrossEntropyLoss(), input_names=["predictions", "labels"], output_names=["ce_loss"]),
        "metrics": BrickMetrics(metric_collection, input_names=["predictions", "labels"]),
    }
    return bricks


def is_equal_model_parameters(model1: torch.nn.Module, model2: torch.nn.Module) -> bool:
    is_equal = []
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        is_equal.append(p1.equal(p2))
    return all(is_equal)
