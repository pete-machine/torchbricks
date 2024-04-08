
from typing import Dict

from torch import nn
from torchbricks.bag_of_bricks import ImageClassifier, Preprocessor
from torchbricks.bricks import BrickCollection, BrickInterface, BrickLoss, BrickMetricSingle, BrickNotTrainable, BrickTrainable
from torchbricks.graph_plotter import create_mermaid_dag_graph
from torchmetrics.classification import MulticlassAccuracy


def test_graph_builder():
    def image_classifier_head(num_classes: int, in_channels: int, input_name: str) -> Dict[str, BrickInterface]:
        """Image classifier bricks: Classifier, loss and metrics """
        head = {
            "classify": BrickTrainable(ImageClassifier(num_classes=num_classes, n_features=in_channels),
                                    input_names=[input_name],
                                    output_names=["./logits", "./probabilities", "./class_prediction"]),
            "accuracy": BrickMetricSingle(MulticlassAccuracy(num_classes=num_classes),
                                        input_names=["./class_prediction", "targets"]),
            "loss": BrickLoss(model=nn.CrossEntropyLoss(),
                            input_names=["./logits", "targets"],
                            output_names=["./loss_ce"])
        }
        return head

    n_features = 5
    bricks = {
        "preprocessor": BrickNotTrainable(Preprocessor(),
                                        input_names=["raw"],
                                        output_names=["normalized"]),
        "backbone": BrickTrainable(nn.Identity(), input_names=["normalized"], output_names=["features"]),
        "head0": image_classifier_head(num_classes=3, in_channels=n_features, input_name="features"),
        "head1": image_classifier_head(num_classes=5, in_channels=n_features, input_name="features"),
    }

    brick_collection = BrickCollection(bricks)
    print(create_mermaid_dag_graph(brick_collection))
    print()
