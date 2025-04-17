from enum import Enum
from typing import Dict, Set

from torchbricks.bricks import Tag

TRAINING: Set[str] = {Tag.MODEL, Tag.LOSS, Tag.METRIC}
VALIDATION: Set[str] = {Tag.MODEL, Tag.LOSS, Tag.METRIC}
TEST: Set[str] = {Tag.MODEL, Tag.LOSS, Tag.METRIC, Tag.METRIC_EXTRA}
EXPORT: Set[str] = {Tag.MODEL}
INFERENCE: Set[str] = {Tag.MODEL}


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
