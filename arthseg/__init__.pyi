from numpy import uint
from numpy.typing import NDArray

def remove_dirt(
    image: NDArray[uint],
    keep: bool = True,
    max_distance: int = 20,
    min_area: float = 0.05,
) -> NDArray[uint]: ...
def fill_holes(
    image: NDArray[uint], fill_value: int, hole_area: float = 0.001
) -> NDArray[uint]: ...
def refine_regions(
    image: NDArray[uint], body_labels: set[int], min_area: float = 0.01
) -> NDArray[uint]: ...
def refine_legs(
    image: NDArray[uint],
    leg_labels: set[int],
    pair_labels: list[tuple[int, int]],
    body_labels: set[int],
    alternative_labels: set[int] = {},
) -> NDArray[uint]: ...
def leg_segments(
    image: NDArray[uint],
    labels: dict[int, list[int]],
    body_labels: set[int],
    alternative_labels: set[int] = {},
) -> NDArray[uint]: ...
