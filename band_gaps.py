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
    """

    bandwidths: np.ndarray
    mid_frequencies: np.ndarray

    @property
    def relative_gaps(self) -> np.ndarray:
        """Divide each bandwidth by its mid-frequency, dimensionless.

        Returns
        -------
        numpy.ndarray
            Relative gap per band pair.

        Raises
        ------
        ValueError
            If any band pair has both of its edges at zero, which leaves its
            relative gap undefined.
        """
        # Only the ratio needs a non-zero centre, so the check lives here
        # rather than blocking the bandwidths a multi-objective run reads
        if np.any(self.mid_frequencies == 0):
            degenerate_pairs = np.flatnonzero(self.mid_frequencies == 0).tolist()
            raise ValueError(
                "A relative gap needs a non-zero mid-frequency; band pairs "
                f"{degenerate_pairs} have both edges at zero."
            )

        return self.bandwidths / self.mid_frequencies


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
        If frequencies is complex, is not a 2-D array holding at least one
        wave number and two bands, or if any entry is not finite or is
        negative.
    """
    frequencies = np.asarray(frequencies)

    # Casting to float64 would drop the imaginary part behind a ComplexWarning,
    # and a complex eigenfrequency is a damped mode, not a rounding artefact
    if np.iscomplexobj(frequencies):
        raise ValueError(
            "frequencies must be real; received a complex array, whose "
            "imaginary part a real band structure cannot carry."
        )

    frequencies = frequencies.astype(np.float64)

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

    return BandGapMetrics(
        bandwidths=upper_edges - lower_edges,
        mid_frequencies=(upper_edges + lower_edges) / 2,
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
        If frequencies fails any check band_gap_metrics makes, or if a band
        pair has both of its edges at zero.
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
    """Apply the single-objective outer function g to one observed cell.

    The acquisition applies g to posterior samples with its own torch maximum;
    this reads the same scalar off measured data, for reporting and for the
    incumbent.

    Parameters
    ----------
    relative_gaps : numpy.ndarray
Relative gap per band a pair of one design, as relative_gap measures it.

    Returns
    -------
    float
        The widest relative gap, the scalar a single-objective run maximizes.

    Raises
    ------
    ValueError
        If relative_gaps is not a 1-D array holding at least one band pair.
    """
    relative_gaps = np.asarray(relative_gaps, dtype=np.float64)

    # A matrix would collapse every design into one number, which reads as a
    # plausible incumbent and is not one
    if relative_gaps.ndim != 1 or relative_gaps.size < 1:
        raise ValueError(
            "relative_gaps must be the 1-D relative gaps of one design, "
            f"holding at least one band pair; received shape "
            f"{tuple(relative_gaps.shape)}."
        )

    return float(np.max(relative_gaps))
