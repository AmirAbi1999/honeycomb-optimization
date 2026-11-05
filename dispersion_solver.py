"""The honeycomb SEM dispersion solver, reached through MATLAB engines.

evaluate_dispersion.m takes its geometry as [hex_angle, cell_width,
face_sheet_thickness, hex_thickness, hex_side_length, hex_side_length], in the
degrees and millimetres a HoneycombGeometry holds, and returns one row of band
frequencies per wave number.

This module contains:
    - SolverSettings
    - solve_dispersion
    - check_matlab
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch import Tensor

from band_gap_metrics import maximum_relative_gap, relative_gap
from geometry import HoneycombGeometry, is_feasible
from matlab_caller import MatlabCaller, MatlabPool

log = logging.getLogger(__name__)

SOURCE_FOLDER = Path(__file__).resolve().parent / "src"

# Every function evaluate_dispersion.m reaches for that lives beside it
SOLVER_FUNCTIONS = (
    "evaluate_dispersion",
    "generate_geometry",
    "geometry_parameters",
    "get_material_property",
    "element_sampling_3D",
    "element_prepare_3D",
    "Mass_and_Stiffness_Element_3D",
)


@dataclass(frozen=True)
class SolverSettings:
    """What the SEM dispersion solver is asked for at every design.

    Attributes
    ----------
    delta : float
        Positive resolution factor of the spectral element mesh.
    minimum_polynomial_order : int
        Element order floor, at least 3.
    n_wave_numbers : int
        Wave numbers swept, at least 2.
    n_bands : int
        Bands returned at each wave number, so n_bands - 1 band pairs.
    cell_width : float
        Unit-cell width held fixed across the run, in millimetres.
    """

    delta: float = 0.22
    minimum_polynomial_order: int = 5
    n_wave_numbers: int = 21
    n_bands: int = 10
    cell_width: float = 1.0


def solve_dispersion(
    caller: MatlabCaller | MatlabPool,
    settings: SolverSettings,
    design: Tensor,
) -> np.ndarray:
    """Solve one design's dispersion relation, refusing an unusable answer.

    Parameters
    ----------
    caller : MatlabCaller or MatlabPool
        Engine or engines with the solver sources on their path. A pool lets
        several threads solve at once.
    settings : SolverSettings
        Mesh, sweep and band counts to solve with.
    design : Tensor
        One design of shape (4,), as [hex_angle, hex_side_length,
        hex_thickness, face_sheet_thickness].

    Returns
    -------
    numpy.ndarray
        Frequencies in hertz, shape (n_wave_numbers, n_bands).

    Raises
    ------
    ValueError
        If the design is not a cell HoneycombGeometry accepts, or if MATLAB
        returns a complex matrix, one of another shape, or one carrying a
        non-finite or negative frequency.
    """
    hex_angle, side_length, hex_thickness, face_sheet = (
        float(value) for value in design
    )
    geometry = HoneycombGeometry(
        hex_angle=hex_angle,
        hex_side_length=side_length,
        hex_thickness=hex_thickness,
        face_sheet_thickness=face_sheet,
        cell_width=settings.cell_width,
    )
    started = time.perf_counter()

    # hex_side_length fills the solver's last two slots, which this project
    # keeps equal
    (frequencies,) = caller.call(
        "evaluate_dispersion",
        [[
            geometry.hex_angle, geometry.cell_width,
            geometry.face_sheet_thickness, geometry.hex_thickness,
            geometry.hex_side_length, geometry.hex_side_length,
        ]],
        settings.delta,
        settings.minimum_polynomial_order,
        settings.n_wave_numbers,
        settings.n_bands,
    )
    log.debug("Solved %s in %.1f s", geometry, time.perf_counter() - started)

    # A damped or unstable mode comes back complex, and casting to float
    # would drop the imaginary part behind a ComplexWarning
    if np.iscomplexobj(frequencies):
        raise ValueError(
            "evaluate_dispersion returned complex frequencies, which mark "
            "damped or unstable modes rather than a real band structure."
        )

    frequencies = frequencies.astype(np.float64)

    expected_shape = (settings.n_wave_numbers, settings.n_bands)
    if frequencies.shape != expected_shape:
        raise ValueError(
            f"evaluate_dispersion returned {frequencies.shape}; expected "
            f"{expected_shape}."
        )

    # A failed eigensolve returns NaN rather than raising, and blaming the
    # band structure downstream would point at the wrong step
    n_broken = int(np.sum(~np.isfinite(frequencies)))
    if n_broken:
        raise ValueError(
            f"evaluate_dispersion returned {n_broken} non-finite frequencies; "
            "the eigensolve did not converge."
        )

    if np.any(frequencies < 0):
        raise ValueError(
            "evaluate_dispersion returned a negative frequency, the lowest "
            f"being {frequencies.min()}."
        )

    return frequencies


def check_matlab(
    bounds: Tensor,
    settings: SolverSettings = SolverSettings(),
    source_folder: str | Path = SOURCE_FOLDER,
) -> np.ndarray:
    """Solve the cell at the centre of the bounds, so a broken setup fails fast.

    Starts its own engine and shuts it down again, so it costs one engine
    start and one solve rather than a whole run.

    Parameters
    ----------
    bounds : Tensor
        Search bounds of shape (2, 4), lower row first, whose centre is solved.
    settings : SolverSettings
        Mesh, sweep and band counts the check solves with.
    source_folder : str or pathlib.Path
        Directory holding the solver sources.

    Returns
    -------
    numpy.ndarray
        Frequencies of the cell it solved, shape (n_wave_numbers, n_bands).

    Raises
    ------
    ValueError
        If the centre of the bounds is not buildable, or if the frequencies
        come back unusable.
    RuntimeError
        If a solver function is missing from the MATLAB path.
    """
    # The centre of the box, so the check exercises a cell the run could
    # actually propose rather than whatever the dataclass defaults hold
    design = bounds.mean(0)
    if not bool(is_feasible(design)):
        raise ValueError(
            f"The centre of the bounds, {design.tolist()}, is not buildable, "
            "so it cannot serve as a check; revise the bounds."
        )

    log.info(
        "Checking the MATLAB solver on design %s",
        np.round(design.numpy(), 4).tolist(),
    )
    started = time.perf_counter()

    with MatlabCaller(source_folder, SOLVER_FUNCTIONS) as caller:
        caller.check()
        frequencies = solve_dispersion(caller, settings, design)

    gaps = relative_gap(frequencies)
    log.info(
        "Solved a %d x %d band structure in %.1f s, %.1f to %.1f Hz",
        *frequencies.shape, time.perf_counter() - started,
        frequencies.min(), frequencies.max(),
    )
    log.info("Relative gaps %s", np.round(gaps, 4).tolist())
    log.info("Widest relative gap %.6f", maximum_relative_gap(gaps))
    log.info("MATLAB check passed")

    return frequencies
