import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class ConcatenatePredictionAndTarget(Metric):
    full_state_update = False  # Maybe this should be False?!
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('predictions', default=[], dist_reduce_fx='cat')
        self.add_state('targets', default=[], dist_reduce_fx='cat')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape[0] == target.shape[0]  # Potentially remove this assertion if it is in your way.

        self.predictions.append(preds)
        self.targets.append(target)

    def compute(self):
        if isinstance(self.predictions, list) and self.predictions:
            predictions = dim_zero_cat(self.predictions)
        else:
            predictions = self.predictions

        if isinstance(self.targets, list) and self.targets:
            targets = dim_zero_cat(self.targets)
        else:
            targets = self.targets

        return predictions, targets
