from typing import Union, Iterable, Tuple, TypedDict, List
from abc import ABC, abstractmethod

from dimos.types.vector import Vector
from dimos.types.path import Path
from dimos.types.costmap import Costmap
from reactivex.observable import Observable
from reactivex.subject import Subject


class VectorDrawConfig(TypedDict, total=False):
    color: str
    width: float
    style: str  # "solid", "dashed", etc.


class PathDrawConfig(TypedDict, total=False):
    color: str
    width: float
    style: str
    fill: bool


class CostmapDrawConfig(TypedDict, total=False):
    colormap: str
    opacity: float
    scale: float


Drawable = Union[
    Vector,
    Path,
    Costmap,
    Tuple[Vector, VectorDrawConfig],
    Tuple[Path, PathDrawConfig],
    Tuple[Costmap, CostmapDrawConfig],
]
Drawables = Iterable[Drawable]


class Visualizable(ABC):
    """
    Base class for objects that can provide visualization data.
    """

    def vis_stream(self) -> Observable[Tuple[str, Drawable]]:
        if not hasattr(self, "_vis_subject"):
            self._vis_subject = Subject()
        return self._vis_subject

    def vis(self, name: str, drawable: Drawable) -> None:
        if not hasattr(self, "_vis_subject"):
            return
        self._vis_subject.on_next((name, drawable))
