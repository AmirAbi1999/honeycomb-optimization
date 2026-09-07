"""One long-lived MATLAB engine driving the SEM dispersion solver.

evaluate_dispersion.m takes its geometry as [hex_angle, cell_width,
face_sheet_thickness, hex_thickness, hex_side_length, hex_side_length], the
same degrees and millimetres a HoneycombGeometry already holds, so the
reordering lives here rather than at each call site.

This module contains:
    - MatlabCaller
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geometry import HoneycombGeometry


class MatlabCaller:
    """Start MATLAB once and reuse it for every geometry evaluation.

    Parameters
    ----------
    source_folder : str or pathlib.Path
        Directory holding evaluate_dispersion.m and the functions it calls.

    Raises
    ------
    FileNotFoundError
        If source_folder is not a directory.
    ImportError
        If the MATLAB engine package is not installed.
    """

    def __init__(self, source_folder: str | Path) -> None:
        folder = Path(source_folder).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"MATLAB source folder not found: {folder}")

        # Imported here, not at module scope, so the rest of the project stays
        # importable on a machine the engine cannot be installed on
        try:
            import matlab
            import matlab.engine
        except ImportError as error:
            raise ImportError(
                "The matlabengine package is required to solve dispersion; it "
                "installs against a MATLAB release and Python 3.13 or older."
            ) from error

        self.matlab = matlab
        self.engine = matlab.engine.start_matlab()

        # A failed addpath would otherwise leave the MATLAB process running
        # with no reference left to quit it
        try:
            self.engine.addpath(str(folder), nargout=0)
        except Exception:
            self.close()
            raise

    def evaluate(
        self,
        geometry: HoneycombGeometry,
        *,
        delta: float,
        minimum_polynomial_order: int,
        n_wave_numbers: int,
        n_bands: int,
    ) -> np.ndarray:
        """Solve one cell's dispersion relation.

        Parameters
        ----------
        geometry : HoneycombGeometry
            Cell to solve, in millimetres and degrees.
        delta : float
            Positive resolution factor of the spectral element mesh.
        minimum_polynomial_order : int
            Element order floor, at least 3 and within the solver's maximum.
        n_wave_numbers : int
            Wave numbers swept, at least 2.
        n_bands : int
            Bands returned at each wave number, MATLAB's n_frequencies.

        Returns
        -------
        numpy.ndarray
            Frequencies in hertz with shape (n_wave_numbers, n_bands).

        Raises
        ------
        ValueError
            If MATLAB returns a matrix of some other shape.
        """
        # hex_side_length fills the solver's last two slots, which this
        # project keeps equal
        geometry_values = [
            geometry.hex_angle,
            geometry.cell_width,
            geometry.face_sheet_thickness,
            geometry.hex_thickness,
            geometry.hex_side_length,
            geometry.hex_side_length,
        ]

        frequencies = self.engine.evaluate_dispersion(
            self.matlab.double([geometry_values]),
            float(delta),
            float(minimum_polynomial_order),
            float(n_wave_numbers),
            float(n_bands),
            nargout=1,
        )
        frequencies = np.asarray(frequencies, dtype=np.float64)

        expected_shape = (n_wave_numbers, n_bands)
        if frequencies.shape != expected_shape:
            raise ValueError(
                f"evaluate_dispersion returned {frequencies.shape}; expected "
                f"{expected_shape}."
            )

        return frequencies

    def close(self) -> None:
        """Shut the engine down, and stay quiet if it is already down."""
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def __enter__(self) -> MatlabCaller:
        """Return the caller with its engine already running."""
        return self

    def __exit__(self, *exception_state: object) -> None:
        """Shut the engine down however the block ended."""
        self.close()
