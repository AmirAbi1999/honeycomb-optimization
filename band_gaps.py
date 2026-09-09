"""Band-gap measurements read off a solved dispersion relation.

Frequencies arrive as one matrix, (n_wave_numbers, n_bands). Bands i and i + 1
bound a pair: its edges are the highest band i and the lowest band
i + 1 over every wave number, and their difference and mean give the bandwidth
and mid-frequency of that pair.

This module contains:
    - BandGapMetrics
    - band_gap_metrics
    - relative_gap
    - bandwidth_mid_frequency
    - maximum_relative_gap
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BandGapMetrics:
    """What one band structure says about each of its band pairs.

    Every array runs from the lowest band pair up and has length
    n_bands - 1, so one index reads the same pair across all three.

    Attributes
    ----------
    bandwidths : numpy.ndarray
        Upper edge - lower edge, in hertz, signed so that a pair whose
        bands overlap keeps a negative width.
    mid_frequencies : numpy.ndarray
        Mean of the two edges, in hertz.
    relative_gaps : numpy.ndarray
        Bandwidth over mid-frequency, dimensionless.
    """

    bandwidths: np.ndarray
    mid_frequencies: np.ndarray
    relative_gaps: np.ndarray


def band_gap_metrics(frequencies: np.ndarray) -> BandGapMetrics:
    """Measure the bandwidth, mid-frequency and relative gap of every band pair.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Eigenfrequencies with shape (n_wave_numbers, n_bands), in hertz.

    Returns
    -------
    BandGapMetrics
        Bandwidths, mid-frequencies and relative gaps of every band pair.

    Raises
    ------
    ValueError
        If frequencies is not a 2-D array holding at least one wave number
        and two bands, if any entry is not finite or is negative, or if a
        band pair has both of its edges at zero.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)

    if frequencies.ndim != 2:
        raise ValueError(
            "frequencies must be 2-D, shaped (n_wave_numbers, n_bands); "
            f"received {frequencies.ndim} dimensions."
        )

    if frequencies.shape[0] < 1 or frequencies.shape[1] < 2:
        raise ValueError(
            "frequencies must hold at least one wave number and two bands to "
            f"form a band pair; received shape {frequencies.shape}."
        )

    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies must all be finite.")

    if np.any(frequencies < 0):
        raise ValueError(
            "frequencies must all be non-negative; the lowest is "
            f"{frequencies.min()}."
        )

    # An eigensolver returns the modes at one wave number in whatever order it
    # converged them, and a band pair only means anything once they ascend
    frequencies = np.sort(frequencies, axis=1)

    # A complete gap has to survive every wave number, so each side of the
    # pair is bounded by the worst one
    lower_edges = np.max(frequencies[:, :-1], axis=0)
    upper_edges = np.min(frequencies[:, 1:], axis=0)

    bandwidths = upper_edges - lower_edges
    mid_frequencies = (upper_edges + lower_edges) / 2

    if np.any(mid_frequencies == 0):
        degenerate_pairs = np.flatnonzero(mid_frequencies == 0).tolist()
        raise ValueError(
            "A relative gap needs a non-zero mid-frequency; band pairs "
            f"{degenerate_pairs} have both edges at zero."
        )

    return BandGapMetrics(
        bandwidths=bandwidths,
        mid_frequencies=mid_frequencies,
        relative_gaps=bandwidths / mid_frequencies,
    )


def relative_gap(frequencies: np.ndarray) -> np.ndarray:
    """Measure the surrogate targets of the single-objective run.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Eigenfrequencies with shape (n_wave_numbers, n_bands), in hertz.

    Returns
    -------
    numpy.ndarray
        Relative gap per band pair, of length n_bands - 1.

    Raises
    ------
    ValueError
        If frequencies fails any check band_gap_metrics makes.
    """
    return band_gap_metrics(frequencies).relative_gaps


def bandwidth_mid_frequency(frequencies: np.ndarray) -> np.ndarray:
    """Measure the surrogate targets of the multi-objective run.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Eigenfrequencies with shape (n_wave_numbers, n_bands), in hertz.

    Returns
    -------
    numpy.ndarray
        Bandwidth and mid-frequency per band pair, interleaved as
        [W1, C1, W2, C2, ...] and of length 2 * (n_bands - 1).

    Raises
    ------
    ValueError
        If frequencies fails any check band_gap_metrics makes.
    """
    metrics = band_gap_metrics(frequencies)

    # Interleave along the last axis: stacking along the first would ravel to
    # every bandwidth followed by every mid-frequency
    return np.stack(
        (metrics.bandwidths, metrics.mid_frequencies), axis=-1,
    ).ravel()


def maximum_relative_gap(relative_gaps: np.ndarray) -> float:
    """Apply the single-objective outer function g to one target vector.

    Parameters
    ----------
    relative_gaps : numpy.ndarray
        Relative gap per band pair, measured by relative_gap or drawn
        from the surrogate posterior.

    Returns
    -------
    float
        The widest relative gap, the scalar a single-objective run maximizes.
    """
    return float(np.max(relative_gaps))
