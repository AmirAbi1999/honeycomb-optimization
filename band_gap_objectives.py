"""Band gaps of a solved dispersion relation, and the BO-CF objectives on them.

Frequencies arrive as (n_wave_numbers, n_bands). Gap i lies between bands i and
i + 1: its edges are the highest band i and the lowest band i + 1 reach over
every wave number. The inner functions h map frequencies to surrogate targets,
and the outer functions g score the targets observed.

This module contains:
    - BandGaps
    - measure_band_gaps
    - relative_widths
    - width_midgap_pairs
    - widest_relative_gap
    - front_points
    - front_hypervolume
    - front_gaps
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor


@dataclass(frozen=True)
class BandGaps:
    """The gap between every pair of adjacent bands, lowest pair first.

    Attributes
    ----------
    lower_edges, upper_edges : numpy.ndarray
        Highest frequency of the lower band and lowest of the upper band over
        every wave number, in hertz, each of length n_bands - 1.
    """

    lower_edges: np.ndarray
    upper_edges: np.ndarray

    @property
    def widths(self) -> np.ndarray:
        """Return each gap's width in hertz, negative where its bands overlap."""
        return self.upper_edges - self.lower_edges

    @property
    def midgaps(self) -> np.ndarray:
        """Return each gap's midgap frequency, in hertz."""
        return (self.upper_edges + self.lower_edges) / 2

    @property
    def relative_widths(self) -> np.ndarray:
        """Divide each width by its midgap frequency.

        Returns
        -------
        numpy.ndarray
            Relative width of every gap, dimensionless.

        Raises
        ------
        ValueError
            If a gap has both of its edges at zero, which leaves its relative
            width undefined.
        """
        midgaps = self.midgaps

        # Only the ratio needs a non-zero midgap, so the check lives here
        # rather than blocking the widths a multi-objective run reads
        if np.any(midgaps == 0):
            raise ValueError(
                "A relative width needs a non-zero midgap; gaps "
                f"{np.flatnonzero(midgaps == 0).tolist()} have both edges at zero."
            )

        return self.widths / midgaps


def measure_band_gaps(frequencies: np.ndarray) -> BandGaps:
    """Find the edges of every gap between adjacent bands.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Eigenfrequencies of shape (n_wave_numbers, n_bands), in hertz.

    Returns
    -------
    BandGaps
        Edges of the n_bands - 1 gaps.
    """
    # An eigensolver returns the modes at one wave number in whatever order it
    # converged them, and a gap only means anything once they ascend
    frequencies = np.sort(frequencies, axis=1)

    # A complete gap has to survive every wave number, so each edge is bounded
    # by the worst one
    return BandGaps(
        lower_edges=frequencies[:, :-1].max(axis=0),
        upper_edges=frequencies[:, 1:].min(axis=0),
    )


def relative_widths(frequencies: np.ndarray) -> np.ndarray:
    """Measure the targets of the single-objective run, its inner function h.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Eigenfrequencies of shape (n_wave_numbers, n_bands), in hertz.

    Returns
    -------
    numpy.ndarray
        Relative width of every gap, of length n_bands - 1.

    Raises
    ------
    ValueError
        If a gap has both of its edges at zero.
    """
    return measure_band_gaps(frequencies).relative_widths


def width_midgap_pairs(frequencies: np.ndarray) -> np.ndarray:
    """Measure the targets of the multi-objective run, its inner function h.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Eigenfrequencies of shape (n_wave_numbers, n_bands), in hertz.

    Returns
    -------
    numpy.ndarray
        Width and midgap of every gap, interleaved as [W1, C1, W2, C2, ...],
        of length 2 * (n_bands - 1).
    """
    gaps = measure_band_gaps(frequencies)

    # Interleaved along the last axis, the layout front_points reads back;
    # stacking along the first would ravel to every width, then every midgap
    return np.stack((gaps.widths, gaps.midgaps), axis=-1).ravel()


def widest_relative_gap(relative_gaps: np.ndarray) -> float:
    """Apply the single-objective outer function g to one observed design.

    The acquisition applies g to posterior samples with its own torch maximum;
    this reads the same scalar off one measured design, for reporting.

    Parameters
    ----------
    relative_gaps : numpy.ndarray
        Relative width of every gap of one design, as relative_widths
        measures them.

    Returns
    -------
    float
        The widest relative gap, the scalar a single-objective run maximizes.

    Raises
    ------
    ValueError
        If relative_gaps is not a 1-D array holding at least one gap.
    """
    relative_gaps = np.asarray(relative_gaps, dtype=np.float64)

    # A matrix would collapse every design into one number, which reads as a
    # plausible incumbent and is not one
    if relative_gaps.ndim != 1 or relative_gaps.size < 1:
        raise ValueError(
            "relative_gaps must be the 1-D relative widths of one design, "
            f"holding at least one gap; received shape {relative_gaps.shape}."
        )

    return float(relative_gaps.max())


def front_points(targets: Tensor, asinh_scale: float | None = None) -> Tensor:
    """Turn interleaved targets into the points of the width-midgap front.

    Parameters
    ----------
    targets : Tensor
        Widths and midgaps interleaved as width_midgap_pairs measures them,
        shape (..., 2 * n_gaps).
    asinh_scale : float, optional
        Hertz at which both axes turn from linear to logarithmic, through
        asinh(value / asinh_scale). None leaves them in hertz.

    Returns
    -------
    Tensor
        One (width, negated midgap) point per gap, shape (..., n_gaps, 2), so
        that both objectives maximize.
    """
    pairs = targets.unflatten(-1, (-1, 2))
    points = torch.stack((pairs[..., 0], -pairs[..., 1]), dim=-1)

    # asinh is odd and increasing, so it keeps every dominance and the sign of
    # the negated midgap, and changes only how much each region weighs
    if asinh_scale is None:
        return points

    return torch.asinh(points / asinh_scale)


def front_hypervolume(
    targets: np.ndarray,
    reference_point: np.ndarray,
    *,
    asinh_scale: float | None = None,
) -> float:
    """Apply the multi-objective outer function g to every design observed.

    Parameters
    ----------
    targets : numpy.ndarray
        Interleaved widths and midgaps of shape (n_designs, 2 * n_gaps), one
        row per design as width_midgap_pairs measures it.
    reference_point : numpy.ndarray
        Two values, as (width, negated midgap) in hertz, bounding the
        dominated region, as the multi-objective acquisition is handed.
    asinh_scale : float, optional
        As front_points takes it.

    Returns
    -------
    float
        Area the gaps dominate, in square hertz or in the asinh units
        asinh_scale sets.

    Raises
    ------
    ValueError
        If targets is not a finite matrix whose columns pair up, or if
        reference_point does not hold exactly two finite values.
    """
    targets = np.asarray(targets, dtype=np.float64)
    reference = np.asarray(reference_point, dtype=np.float64)

    if (
        targets.ndim != 2
        or targets.shape[0] < 1
        or targets.shape[1] < 2
        or targets.shape[1] % 2 != 0
        or not np.all(np.isfinite(targets))
    ):
        raise ValueError(
            "targets must be a finite matrix of one row per design, "
            "interleaving one width and one midgap per gap; received shape "
            f"{targets.shape}."
        )

    if reference.shape != (2,) or not np.all(np.isfinite(reference)):
        raise ValueError(
            "reference_point must hold exactly two finite values, a width and "
            f"a negated midgap; received {reference.tolist()}."
        )

    points = front_points(torch.as_tensor(targets), asinh_scale).reshape(-1, 2)
    if asinh_scale is not None:
        reference = np.arcsinh(reference / asinh_scale)

    # A point short of the reference on either side dominates nothing, and
    # clamping it there zeroes its share without a separate filter
    points = np.maximum(points.numpy(), reference)

    # Widest first, so each point adds only the height it reaches above every
    # wider point before it, and the dominated staircase is measured exactly
    ordered = points[np.argsort(-points[:, 0], kind="stable")]
    heights = np.maximum.accumulate(ordered[:, 1])
    steps = np.diff(heights, prepend=reference[1])

    return float(np.sum((ordered[:, 0] - reference[0]) * steps))


def front_gaps(targets: Tensor, reference_point: np.ndarray) -> np.ndarray:
    """List the gaps on the front, lowest midgap first.

    Parameters
    ----------
    targets : Tensor
        Interleaved widths and midgaps of shape (n_designs, 2 * n_gaps), as
        width_midgap_pairs measures them.
    reference_point : numpy.ndarray
        Two values, as (width, negated midgap), a gap has to beat to count.

    Returns
    -------
    numpy.ndarray
        One row per gap on the front, as (design index, gap index, width,
        midgap), both indices counting from zero.
    """
    points = front_points(torch.as_tensor(targets))
    n_gaps = points.shape[-2]
    points = points.reshape(-1, 2)
    reference = torch.as_tensor(reference_point, dtype=points.dtype)

    # A gap short of the reference dominates nothing, and cannot be dominated
    # away by one that also falls short
    on_front = is_non_dominated(points) & (points > reference).all(-1)
    indices = on_front.nonzero().squeeze(-1)
    rows = np.column_stack((
        (indices // n_gaps).numpy(),
        (indices % n_gaps).numpy(),
        points[indices, 0].numpy(),
        -points[indices, 1].numpy(),
    ))

    return rows[np.argsort(rows[:, 3])]
