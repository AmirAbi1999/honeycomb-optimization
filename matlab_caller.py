"""One long-lived MATLAB engine driving the SEM dispersion solver.

evaluate_dispersion.m takes its geometry as [hex_angle, cell_width,
face_sheet_thickness, hex_thickness, hex_side_length, hex_side_length], the
same degrees and millimetres a HoneycombGeometry already holds, so the
reordering lives here rather than at each call site.

This module contains:
    - MatlabCaller
    - MatlabPool
"""

from __future__ import annotations

import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from geometry import HoneycombGeometry

log = logging.getLogger(__name__)

# Every function evaluate_dispersion.m reaches for that lives beside it
REQUIRED_FUNCTIONS = (
    "evaluate_dispersion",
    "generate_geometry",
    "geometry_parameters",
    "get_material_property",
    "element_sampling_3D",
    "element_prepare_3D",
    "Mass_and_Stiffness_Element_3D",
)


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

        # Starting the engine costs tens of seconds, long enough that a silent
        # wait reads as a hang
        log.info("Starting the MATLAB engine, which takes a moment")
        started = time.perf_counter()
        self.engine = matlab.engine.start_matlab()
        log.info(
            "MATLAB %s ready in %.1f s",
            self.engine.version(), time.perf_counter() - started,
        )

        # A failed addpath would otherwise leave the MATLAB process running
        # with no reference left to quit it
        try:
            self.engine.addpath(str(folder), nargout=0)
        except Exception:
            self.close()
            raise

        log.info("Added the solver sources to the MATLAB path: %s", folder)

    def check(self) -> None:
        """Confirm the engine can see every function a solve needs.

        Raises
        ------
        RuntimeError
            If any required function is not on the engine's path, which
            addpath reports no error for.
        """
        missing = [
            name for name in REQUIRED_FUNCTIONS
            if int(self.engine.exist(name, "file")) != 2
        ]
        if missing:
            raise RuntimeError(
                "The MATLAB path is missing "
                f"{', '.join(missing)}; check the source folder."
            )

        log.info("All %d solver functions are on the MATLAB path", len(
            REQUIRED_FUNCTIONS
        ))

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
            If MATLAB returns a matrix of some other shape, a complex matrix,
            or one carrying a non-finite or negative frequency.
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

        log.debug("Solving %s", geometry_values)
        started = time.perf_counter()

        frequencies = self.engine.evaluate_dispersion(
            self.matlab.double([geometry_values]),
            float(delta),
            float(minimum_polynomial_order),
            float(n_wave_numbers),
            float(n_bands),
            nargout=1,
        )
        frequencies = np.asarray(frequencies)

        log.debug("Solved in %.1f s", time.perf_counter() - started)

        # A damped or unstable mode comes back complex, and casting to float
        # would drop the imaginary part behind a ComplexWarning
        if np.iscomplexobj(frequencies):
            raise ValueError(
                "evaluate_dispersion returned complex frequencies, which mark "
                "damped or unstable modes rather than a real band structure."
            )

        frequencies = frequencies.astype(np.float64)

        expected_shape = (n_wave_numbers, n_bands)
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
                f"evaluate_dispersion returned {n_broken} non-finite "
                "frequencies; the eigensolve did not converge."
            )

        if np.any(frequencies < 0):
            raise ValueError(
                "evaluate_dispersion returned a negative frequency, the "
                f"lowest being {frequencies.min()}."
            )

        return frequencies

    def close(self) -> None:
        """Shut the engine down, and stay quiet if it is already down."""
        if self.engine is not None:
            self.engine.quit()
            self.engine = None
            log.info("MATLAB engine shut down")

    def __enter__(self) -> MatlabCaller:
        """Return the caller with its engine already running."""
        return self

    def __exit__(self, *exception_state: object) -> None:
        """Shut the engine down however the block ended."""
        self.close()


class MatlabPool:
    """One engine per worker, checked out for the length of a single solve.

    Answers evaluate with the same signature MatlabCaller does, so a caller
    that solves one design at a time cannot tell the two apart. An engine
    serves one call at a time, so concurrency comes from holding several.

    Parameters
    ----------
    source_folder : str or pathlib.Path
        Directory holding evaluate_dispersion.m and the functions it calls.
    n_workers : int
        Engines to start, and so solves that can be in flight at once.

    Raises
    ------
    ValueError
        If n_workers is not positive.
    """

    def __init__(self, source_folder: str | Path, n_workers: int) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be positive; received {n_workers}.")

        log.info("Starting %d MATLAB engines in parallel", n_workers)
        started = time.perf_counter()

        # Startup blocks in the engine's C extension rather than on the GIL,
        # so N engines cost about as long as one rather than N times as long
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(MatlabCaller, source_folder) for _ in range(n_workers)
            ]
            self.callers: list[MatlabCaller] = []
            try:
                for future in futures:
                    self.callers.append(future.result())
            except Exception:
                # Whatever started has to be shut down, or the failed run
                # leaves orphaned MATLAB processes behind
                for future in futures:
                    if not future.cancelled() and future.exception() is None:
                        future.result().close()
                raise

        log.info(
            "%d MATLAB engines ready in %.1f s",
            n_workers, time.perf_counter() - started,
        )

        self._free: queue.Queue[MatlabCaller] = queue.Queue()
        for caller in self.callers:
            self._free.put(caller)

    def check(self) -> None:
        """Confirm every engine can see the functions a solve needs.

        Raises
        ------
        RuntimeError
            If any engine is missing a required function.
        """
        for caller in self.callers:
            caller.check()

    def evaluate(self, *arguments: object, **keywords: object) -> np.ndarray:
        """Solve one cell on whichever engine is free, waiting for one.

        Parameters
        ----------
        *arguments, **keywords
            Passed through to MatlabCaller.evaluate unchanged.

        Returns
        -------
        numpy.ndarray
            Frequencies in hertz, as MatlabCaller.evaluate returns them.
        """
        caller = self._free.get()
        try:
            return caller.evaluate(*arguments, **keywords)
        finally:
            self._free.put(caller)

    def close(self) -> None:
        """Shut every engine down, and stay quiet about the ones already down."""
        for caller in self.callers:
            caller.close()
        self.callers = []

    def __enter__(self) -> MatlabPool:
        """Return the pool with every engine already running."""
        return self

    def __exit__(self, *exception_state: object) -> None:
        """Shut every engine down however the block ended."""
        self.close()
