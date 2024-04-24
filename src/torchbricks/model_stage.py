from enum import Enum
from typing import Dict, Set

from torchbricks import brick_group

TRAINING: Set[str] = {brick_group.MODEL, brick_group.LOSS, brick_group.METRIC}
VALIDATION: Set[str] = {brick_group.MODEL, brick_group.LOSS, brick_group.METRIC}
TEST: Set[str] = {brick_group.MODEL, brick_group.LOSS, brick_group.METRIC, brick_group.METRIC_EXTRA}
EXPORT: Set[str] = {brick_group.MODEL}
INFERENCE: Set[str] = {brick_group.MODEL}


class ModelStage(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    INFERENCE = "inference"
    EXPORT = "export"


DEFAULT_MODEL_STAGE_GROUPS: Dict[str, Set[str]] = {
    ModelStage.TRAIN.value: TRAINING,
    ModelStage.VALIDATION.value: VALIDATION,
    ModelStage.TEST.value: TEST,
    ModelStage.EXPORT.value: EXPORT,
    ModelStage.INFERENCE.value: INFERENCE,
}
