from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from PIL import ImageColor
from typeguard import typechecked


class Task(Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    # OBJECT_DETECTION = "object_detection"
    # POSE_ESTIMATION = "pose_estimation"


@dataclass
class TaskInfo:
    pass


@dataclass
class ImageClassificationInfo(TaskInfo):
    label_name: str
    class_names: List[str]
    class_ids: List[int]
    class_colors: List  # Type is "List[Tuple[int, int, int]]" but is not supported by hydra/omegaconf

    @classmethod
    def from_names(cls, class_names: List[str]) -> "ImageClassificationInfo":
        n_classes = len(class_names)
        return cls(
            label_name="label", class_names=class_names, class_ids=list(range(n_classes)), class_colors=get_distinct_colors(n_classes)
        )


class ImageSegmentationInfo(ImageClassificationInfo):
    pass


@typechecked
def get_distinct_colors(n_colors: int) -> List[Tuple[int, int, int]]:
    if n_colors > len(distinct_colors):
        raise ValueError(f"Only {len(distinct_colors)} distinct colors are available. Consider supporting more colors.")

    return [ImageColor.getrgb(color) for color in distinct_colors[:n_colors]]


distinct_colors = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
]
