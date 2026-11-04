"""Long-lived MATLAB engines, alone or pooled, calling into one source folder.

A caller names a source folder and the functions it needs, and gets numpy
arrays back.

This module contains:
    - MatlabCaller
    - MatlabPool
"""

from __future__ import annotations

import logging
import queue
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class MatlabCaller:
    """Start MATLAB once, put a source folder on its path, and call into it.

    Parameters
    ----------
    source_folder : str or pathlib.Path
        Directory holding the MATLAB functions to call.
    required_functions : sequence of str
        Functions check confirms are on the path.

    Raises
    ------
    FileNotFoundError
        If source_folder is not a directory.
    ImportError
        If the MATLAB engine package is not installed.
    """

    def __init__(
        self,
        source_folder: str | Path,
        required_functions: Sequence[str] = (),
    ) -> None:
        folder = Path(source_folder).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"MATLAB source folder not found: {folder}")

        # Imported here, not at module scope, so the rest of a project stays
        # importable on a machine the engine cannot be installed on
        try:
            import matlab
            import matlab.engine
        except ImportError as error:
            raise ImportError(
                "The matlabengine package is required to call MATLAB; it "
                "installs against a MATLAB release and Python 3.13 or older."
            ) from error

        self.matlab = matlab
        self.required_functions = tuple(required_functions)

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

        log.info("Added the source folder to the MATLAB path: %s", folder)

    def check(self) -> None:
        """Confirm the engine can see every required function.

        Raises
        ------
        RuntimeError
            If a required function is not on the engine's path, which addpath
            reports no error for.
        """
        missing = [
            name for name in self.required_functions
            if int(self.engine.exist(name, "file")) != 2
        ]
        if missing:
            raise RuntimeError(
                f"The MATLAB path is missing {', '.join(missing)}; check the "
                "source folder."
            )

        log.info(
            "All %d required functions are on the MATLAB path",
            len(self.required_functions),
        )

    def call(
        self,
        function_name: str,
        *arguments: object,
        nargout: int = 1,
    ) -> tuple[np.ndarray, ...]:
        """Call one MATLAB function and return its outputs as numpy arrays.

        Parameters
        ----------
        function_name : str
            Function on the engine's path.
        *arguments : object
            Strings, passed as they are, or real numbers, arrays and nested
            lists, sent as doubles.
        nargout : int
            Outputs to ask MATLAB for.

        Returns
        -------
        tuple of numpy.ndarray
            One array per output, nargout of them.
        """
        outputs = getattr(self.engine, function_name)(
            *(self._to_matlab(argument) for argument in arguments),
            nargout=nargout,
        )

        # MATLAB hands back None for no output and a bare value for one, so
        # every count comes back as one tuple and callers never branch on it
        if nargout < 2:
            outputs = () if nargout == 0 else (outputs,)

        return tuple(np.asarray(output) for output in outputs)

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

    def _to_matlab(self, value: object) -> object:
        """Send a string as it is, a number as a double, an array as a matrix."""
        if isinstance(value, str):
            return value

        # MATLAB reads a Python int as int64, which most numerical code cannot
        # mix with doubles, so integers go as doubles too
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            return float(array)

        return self.matlab.double(array.tolist())


class MatlabPool:
    """One engine per worker, lent to a single call at a time.

    Answers check and call as a MatlabCaller does, so code that makes one call
    at a time cannot tell the two apart. An engine serves one call at a time,
    so concurrency comes from holding several.

    Parameters
    ----------
    source_folder : str or pathlib.Path
        Directory holding the MATLAB functions to call.
    n_workers : int
        Engines to start, and so calls that can be in flight at once.
    required_functions : sequence of str
        Functions check confirms are on every engine's path.

    Raises
    ------
    ValueError
        If n_workers is not positive.
    """

    def __init__(
        self,
        source_folder: str | Path,
        n_workers: int,
        required_functions: Sequence[str] = (),
    ) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be positive; received {n_workers}.")

        log.info("Starting %d MATLAB engines in parallel", n_workers)
        started = time.perf_counter()

        # Startup blocks in the engine's C extension rather than on the GIL,
        # so N engines cost about as long as one rather than N times as long
        start = partial(MatlabCaller, source_folder, required_functions)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(start) for _ in range(n_workers)]
            try:
                self.callers = [future.result() for future in futures]
            except Exception:
                # Whatever started has to be shut down, or the failure leaves
                # orphaned MATLAB processes behind
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
        """Confirm every engine can see every required function.

        Raises
        ------
        RuntimeError
            If any engine is missing a required function.
        """
        for caller in self.callers:
            caller.check()

    def call(
        self,
        function_name: str,
        *arguments: object,
        nargout: int = 1,
    ) -> tuple[np.ndarray, ...]:
        """Call one MATLAB function on whichever engine is free, waiting for one.

        Parameters
        ----------
        function_name, *arguments, nargout
            As MatlabCaller.call takes them.

        Returns
        -------
        tuple of numpy.ndarray
            One array per output, as MatlabCaller.call returns them.
        """
        caller = self._free.get()
        try:
            return caller.call(function_name, *arguments, nargout=nargout)
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
