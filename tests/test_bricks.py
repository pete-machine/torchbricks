from typing import Optional

import pytest
import torch
import torchmetrics
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from torchbricks.bag_of_bricks import custom_metrics
from torchbricks.brick_collection import BrickCollection
from torchbricks.bricks import (
    BrickLoss,
    BrickMetrics,
    BrickMetricSingle,
)


@pytest.mark.parametrize("metric_name", ("accuracy", None))
def test_brick_torch_metric_single_metric(metric_name: Optional[str]):
    num_classes = 5
    bricks = {
        "accuracy": BrickMetricSingle(
            MulticlassAccuracy(num_classes=num_classes), input_names=["logits", "targets"], metric_name=metric_name
        ),
        "loss": BrickLoss(model=nn.CrossEntropyLoss(), input_names=["logits", "targets"], output_names=["loss_ce"]),
    }

    model = BrickCollection(bricks)

    batch_logits = torch.rand((1, num_classes))
    named_inputs = {"logits": batch_logits, "targets": torch.ones((1), dtype=torch.int64)}
    model(named_inputs=named_inputs)
    metrics = model.summarize(reset=True)
    expected_metric_name = metric_name or "MulticlassAccuracy"

    assert list(metrics) == [expected_metric_name]


@pytest.mark.parametrize("return_metrics", [False, True])
def test_brick_torch_metric_multiple_metric(return_metrics: bool):
    num_classes = 5
    metric_collection = torchmetrics.MetricCollection(
        {
            "MeanAccuracy": MulticlassAccuracy(num_classes=num_classes, average="macro"),
            "Accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
            "ConfMat": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes),
            "Concatenate": custom_metrics.ConcatenatePredictionAndTarget(compute_on_cpu=True),
        }
    )

    bricks = {
        "metrics": BrickMetrics(metric_collection, input_names=["logits", "targets"], return_metrics=return_metrics),
        "loss": BrickLoss(model=nn.CrossEntropyLoss(), input_names=["logits", "targets"], output_names=["loss_ce"]),
    }

    model = BrickCollection(bricks)
    batch_logits = torch.rand((1, num_classes))
    named_inputs = {"logits": batch_logits, "targets": torch.ones((1), dtype=torch.int64)}
    named_outputs = model(named_inputs=named_inputs)

    expected_outputs = {"logits", "targets", "loss_ce"}
    if return_metrics:
        expected_outputs = expected_outputs.union(set(metric_collection))

    assert set(named_outputs) == expected_outputs
    metrics = model.summarize(reset=True)
    expected_metrics = set(metric_collection)
    assert set(metrics) == expected_metrics
