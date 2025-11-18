import logging

import numpy as np
import numpy.typing as npt

from .types import Cover, CoverScheme

__all__ = [
    "precomputed_cover",
    "rectangular_cover",
    "Width_Balanced_Cover",
    "Data_Balanced_Cover",
]

logger = logging.getLogger("zen_mapper")


def precomputed_cover(cover: Cover) -> CoverScheme:
    """A precomputed cover

    Parameters
    ----------
    cover : Cover
        the precomputed cover to use
    """

    def inner(*_):
        return cover

    return inner  # type: ignore


def rectangular_cover(centers, widths, data, tol=1e-9):
    if len(centers.shape) == 1:
        centers = centers.reshape(-1, 1)

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    distances = np.abs(data - centers[:, None])
    return list(
        map(
            np.flatnonzero,
            np.all(
                distances * 2 - widths <= tol,
                axis=2,
            ),
        )
    )


def _grid(start, stop, steps):
    """Create an n-dimensional grid from start to stop with steps

    Parameters
    ----------
    start : ndarray
        The point to start at
    stop : ndarray
        The point to stop at
    steps : int | ndarray
        The number of grid points for each direction

    Raises
    ------

    ValueError
        If len(start) != len(stop)
    """

    if len(start) != len(stop):
        raise ValueError("Start and stop points need to have same dimension")

    dims = (
        np.linspace(begin, end, num=num)
        for begin, end, num in np.broadcast(start, stop, steps)
    )
    grid = np.meshgrid(*dims)
    return np.column_stack([dim.reshape(-1) for dim in grid])


class Width_Balanced_Cover:
    """A cover comprised of equally sized rectangular elements

    Parameters
    ----------
    n_elements : ArrayLike
        the number of covering elements along each dimension. If the data is
        dimension d this results in d^n covering elements.

    percent_overlap : float
        a number between 0 and 1 representing the ammount of overlap between
        adjacent covering elements.


    Raises
    ------
    Value Error
        if n_elements < 1
    Value Error
        if percent_overlap is not in (0,1)
    """

    def __init__(self, n_elements: npt.ArrayLike, percent_overlap: float):
        n_elements = np.array([n_elements], dtype=int)

        if np.any(n_elements < 1):
            raise ValueError("n_elements must be at least 1")

        if not 0 < percent_overlap < 1:
            raise ValueError("percent_overlap must be in the range (0,1)")

        self.n_elements = n_elements
        self.percent_overlap = percent_overlap

    def __call__(self, data):
        logger.info("Computing the width balanced cover")

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        upper_bound = np.max(data, axis=0).astype(float)
        lower_bound = np.min(data, axis=0).astype(float)

        width = (upper_bound - lower_bound) / (
            self.n_elements - (self.n_elements - 1) * self.percent_overlap
        )
        width = width.flatten()
        self.width = width

        # Compute the centers of the "lower left" and "upper right" cover
        # elements
        upper_bound -= width / 2
        lower_bound += width / 2

        centers = _grid(lower_bound, upper_bound, self.n_elements)
        self.centers = centers
        return rectangular_cover(centers, width, data)


class Data_Balanced_Cover:
    """A cover constructed by dividing data into intervals which contain
    roughly the same number of datapoints.

    Description
    -----------
    The cover is constructed by first applying the width-balanced cover
    over sorted index positions `[0, ..., N-1]`
    and then mapping those indexed cover regions back.
    If there are `n_elements = k` bins across `N` sorted points,
    each bin has a base size:

        `base_size = N / (k - (k - 1)*overlap)`

        `step = base_size * (1 - overlap)`

    The first cover element spans indices `[0, base_size)`,
    the next starts at `step`, and so on, creating overlapping intervals
    such that `percent_overlap = 0.5` means that each interval shares 50% of
    its points with the next.

    Parameters
    ----------
    n_elements : int
        Number of cover elements to create.
    percent_overlap : float
        Fractional overlap between adjacent cover elements, with respect
        to the number of data points.

    Raises
    ------
    ValueError
        If n_elements < 1
    ValueError
        If percent_overlap not in (0, 1)
    ValueError
        If projection has dimension < 1
    ValueError
        If number of data points < n_elements
    """

    def __init__(self, n_elements: int, percent_overlap: float):
        if n_elements < 1:
            raise ValueError("n_elements must be at least 1")
        if not 0 < percent_overlap < 1:
            raise ValueError("percent_overlap must be in the range (0,1)")

        self.n_elements = int(n_elements)
        self.percent_overlap = percent_overlap

    def __call__(self, data: npt.ArrayLike):
        data = np.asarray(data, dtype=float)

        if data.ndim != 1:
            raise ValueError(
                f"Data_Balanced_Cover only supports 1-dimensional input"
                f"(projected) data but received data with dim: {data.ndim}"
            )

        logger.info("Computing the data balanced cover")

        n = len(data)
        if n < self.n_elements:
            raise ValueError("Number of data points must be >= n_elements")

        sort_idx = np.argsort(data)

        idxs = np.arange(n)
        cover_idxs = self._width_balanced_cover_indices(idxs)

        cover = [sort_idx[g] for g in cover_idxs]

        return cover

    def _width_balanced_cover_indices(self, indices: np.ndarray):
        """Build width-balanced overlapping intervals over 1D indices."""
        n = len(indices)
        k = self.n_elements

        base_size = n / (k - (k - 1) * self.percent_overlap)
        step = base_size * (1 - self.percent_overlap)

        covers = []
        for i in range(k):
            start = int(round(i * step))
            end = int(round(start + base_size))
            grp = np.arange(max(0, start), min(n, end))
            covers.append(grp)
        return covers
