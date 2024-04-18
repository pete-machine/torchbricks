from typing import Dict, Set

from torchbricks import brick_group

TRAINING: Set[str] = {brick_group.MODEL, brick_group.LOSS, brick_group.METRIC}
VALIDATION: Set[str] = {brick_group.MODEL, brick_group.LOSS, brick_group.METRIC}
TEST: Set[str] = {brick_group.MODEL, brick_group.LOSS, brick_group.METRIC, brick_group.METRIC_EXTRA}
EXPORT: Set[str] = {brick_group.MODEL}
INFERENCE: Set[str] = {brick_group.MODEL}

# class ModelStage(Enum):
#     TRAIN = "train"
#     VALIDATION = "validation"
#     TEST = "test"
#     INFERENCE = "inference"


DEFAULT_MODEL_STAGE_GROUPS: Dict[str, Set[str]] = {
    "TRAIN": TRAINING,
    "VALIDATION": VALIDATION,
    "TEST": TEST,
    "EXPORT": EXPORT,
    "INFERENCE": INFERENCE,
}
