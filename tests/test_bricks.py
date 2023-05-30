from torch_bricks import bricks
from torch_bricks.bricks import RunState
from torch_bricks import custom_metrics
from typing import Dict
import torch
from torch import nn
import torchmetrics
from torchmetrics import classification

def create_brick_collection(num_classes: int, num_backbone_featues: int) -> Dict[str, bricks.Brick]:
    class Preprocessor(nn.Module):
        def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
            return raw_input/2

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


    metric_collection = torchmetrics.MetricCollection({
        'MeanAccuracy': classification.MulticlassAccuracy(num_classes=num_classes, average='macro', multiclass=True),
        'Accuracy': classification.MulticlassAccuracy(num_classes=num_classes, average='micro', multiclass=True),
        'ConfMat': torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes),
        'Concatenate': custom_metrics.ConcatenatePredictionAndTarget(compute_on_cpu=True)
    })
    brick_collections = {
        'preprocessor': bricks.BrickNotTrainable(Preprocessor(), input_names=['raw'], output_names=['preprocessed']),
        'backbone': bricks.BrickTrainable(TinyBackbone(n_kernels=num_backbone_featues), input_names=['preprocessed'],
                                                output_names=['features']),
        'classifier': bricks.BrickTrainable(Classifier(input_channels=num_backbone_featues, num_classes=num_classes),
                                            input_names=['features'],
                                            output_names=['predictions']),
        'loss': bricks.BrickLoss(nn.CrossEntropyLoss(), input_names=['predictions', 'labels'], output_names=['ce_loss']),
        'metrics': bricks.BrickTorchMetric(metric_collection, input_names=['predictions', 'labels'])
    }
    return brick_collections


def test_brick_collection():
    num_classes = 10
    num_backbone_featues = 5

    brick_collection = create_brick_collection(num_classes=num_classes, num_backbone_featues=num_backbone_featues)

    model = bricks.BrickCollection(bricks=brick_collection)
    named_tensors = {'labels': torch.tensor(range(num_classes), dtype=torch.float64), 'raw': torch.zeros((3, 24, 24))}

    state = RunState.TRAIN
    model(state=state, named_tensors=named_tensors)
    model.on_step(state=state, named_tensors=named_tensors, batch_idx=0)
    model.on_step(state=state, named_tensors=named_tensors, batch_idx=0)
    model.on_step(state=state, named_tensors=named_tensors, batch_idx=0)
    model.summarize(state=state, reset=True)
