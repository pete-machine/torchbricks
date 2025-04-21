from enum import Enum
from typing import Dict, Set


class Tag:
    MODEL = "MODEL"
    LOSS = "LOSS"
    METRIC = "METRIC"
    METRIC_EXTRA = "METRIC_EXTRA"
    VISUALIZATION = "VISUALIZATION"


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


DEFAULT_MODEL_STAGE_TAGS: Dict[str, Set[str]] = {
    ModelStage.TRAIN.name: TRAINING,
    ModelStage.VALIDATION.name: VALIDATION,
    ModelStage.TEST.name: TEST,
    ModelStage.EXPORT.name: EXPORT,
    ModelStage.INFERENCE.name: INFERENCE,
}
