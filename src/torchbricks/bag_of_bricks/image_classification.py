from typing import Dict

import torch
import torchmetrics
from torch import nn
from torchbricks.bag_of_bricks.task_utils import TaskInfo
from torchbricks.bricks import BrickInterface, BrickLoss, BrickMetrics, BrickTrainable
from torchmetrics import classification


def create_image_classification_head(n_backbone_features: int, task_info: TaskInfo, input_name: str) -> Dict[str, BrickInterface]:
    num_classes = len(task_info.class_names)
    target_name = task_info.label_name
    metrics = torchmetrics.MetricCollection(
        {
            "MeanAccuracy": classification.MulticlassAccuracy(num_classes=num_classes, average="macro", multiclass=True),
            "Accuracy": classification.MulticlassAccuracy(num_classes=num_classes, average="micro", multiclass=True),
            "ConfMat": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes),
            # "Concatenate": ConcatenatePredictionAndTarget(compute_on_cpu=True),
        }
    )
    bricks = {
        "classifier": BrickTrainable(
            ImageClassifier(num_classes=num_classes, n_features=n_backbone_features),
            input_names=[input_name],
            output_names=["./logits", "./probabilities", "./class_prediction"],
        ),
        "loss": BrickLoss(model=nn.CrossEntropyLoss(), input_names=["./logits", target_name], output_names=["./loss"]),
        "metrics": BrickMetrics(metric_collection=metrics, input_names=["./class_prediction", target_name]),
    }
    return bricks


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
