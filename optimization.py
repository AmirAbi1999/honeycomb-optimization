"""Single-objective BO-CF run over honeycomb unit cells.

One round fits a surrogate per band pair to the relative gaps measured so far,
maximizes qLogEI of their composite maximum over the buildable box, and solves
the n_workers cells it proposes on one MATLAB engine each. A design carries the
four parameters the geometry constraints read, in the order constraint_margins
expects::

    design = [hex_angle, hex_side_length, hex_thickness, face_sheet_thickness]

cell_width is held at the value SolverSettings carries, since neither
constraint reads it.

This module contains:
    - SolverSettings
    - RunConfig
    - RunHistory
    - matlab_evaluator
    - check_matlab
    - run_single_objective
    - main
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from acquisition import propose_batch, single_objective_acquisition
from band_gaps import maximum_relative_gap, relative_gap
from bocf_models import SurrogateConfig, build_surrogates, fit_surrogates
from geometry import HoneycombGeometry, is_feasible
from matlab_caller import MatlabCaller, MatlabPool

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

# Centred on the defaults geometry_parameters.m carries, and wide enough that
# a Sobol draw clears both constraints about four times in five
DESIGN_BOUNDS = torch.tensor([
    [70.0, 4.0, 0.5, 0.4],
    [165.0, 10.0, 2.0, 1.5],
], dtype=torch.double)


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


@dataclass(frozen=True)
class RunConfig:
    """How one single-objective run is paced.

    Attributes
    ----------
    bounds : Tensor
        Search bounds of shape (2, 4), lower row first.
    n_initial : int
        Buildable Sobol designs solved before the first surrogate is fitted.
    n_iterations : int
        Fit-and-propose rounds after that, each solving n_workers designs, so
        the run costs n_initial + n_iterations * n_workers solves in total.
    n_workers : int
        Solves in flight at once, and so the designs one round proposes.
    clearance : float
        Margin a proposal keeps in hand, passed through to propose_design.
    max_failures : int
        Consecutive rounds losing every proposal before the run gives up.
    seed : int
        Fixes the initial draw; each round offsets it by its own index.
    surrogate : SurrogateConfig
        Covariance the per-pair surrogates are built with.
    """

    # Cloned per run, so one config cannot edit the bounds of another
    bounds: Tensor = field(default_factory=DESIGN_BOUNDS.clone)
    n_initial: int = 40
    n_iterations: int = 45
    n_workers: int = 8
    clearance: float = 1e-6
    max_failures: int = 5
    seed: int = 0
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)

    def __post_init__(self) -> None:
        """Reject a budget that cannot start or cannot propose.

        Raises
        ------
        ValueError
            If n_workers is not positive, or if fewer than two initial
            designs are asked for, which leaves nothing to fit.
        """
        if self.n_workers < 1:
            raise ValueError(
                f"n_workers must be positive; received {self.n_workers}."
            )

        if self.n_initial < 2:
            raise ValueError(
                "n_initial must be at least 2, or no surrogate can be fitted; "
                f"received {self.n_initial}."
            )


@dataclass(frozen=True)
class RunHistory:
    """Everything one run measured, in the order it measured it.

    Attributes
    ----------
    designs : Tensor
        Solved designs of shape (n_solved, 4).
    relative_gaps : Tensor
        Relative gap per band pair, shape (n_solved, n_bands - 1).
    incumbents : numpy.ndarray
        Best relative gap known after each solve, of length n_solved.
    """

    designs: Tensor
    relative_gaps: Tensor
    incumbents: np.ndarray

    @property
    def best_design(self) -> Tensor:
        """Return the design holding the widest relative gap seen."""
        return self.designs[int(np.argmax(self.incumbents))]

    @property
    def best_relative_gap(self) -> float:
        """Return the widest relative gap seen."""
        return float(self.incumbents[-1])


def matlab_evaluator(
    caller: MatlabCaller | MatlabPool,
    settings: SolverSettings,
) -> Callable[[Tensor], np.ndarray]:
    """Adapt a MATLAB engine, or a pool of them, into the run loop's evaluator.

    Parameters
    ----------
    caller : MatlabCaller or MatlabPool
        Engine or engines already pointed at the solver sources. A pool makes
        the returned evaluator safe to call from several threads at once.
    settings : SolverSettings
        Mesh, sweep and band counts every design is solved with.

    Returns
    -------
    callable
        Maps a design of shape (4,) to frequencies of shape
        (n_wave_numbers, n_bands).
    """
    def evaluate(design: Tensor) -> np.ndarray:
        """Solve the dispersion relation of one design."""
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

        return caller.evaluate(
            geometry,
            delta=settings.delta,
            minimum_polynomial_order=settings.minimum_polynomial_order,
            n_wave_numbers=settings.n_wave_numbers,
            n_bands=settings.n_bands,
        )

    return evaluate


def check_matlab(
    settings: SolverSettings = SolverSettings(),
    source_folder: str | Path | None = None,
    bounds: Tensor | None = None,
) -> np.ndarray:
    """Solve one cell end to end, so a broken setup fails in seconds.

    Starts its own engine and shuts it down again, so it costs one engine
    start and one solve rather than a whole run.

    Parameters
    ----------
    settings : SolverSettings
        Mesh, sweep and band counts the check solves with.
    source_folder : str or pathlib.Path, optional
        Directory holding the solver sources, ROOT / "src" by default.
    bounds : Tensor, optional
        Box whose centre is solved, the run's own bounds by default.

    Returns
    -------
    numpy.ndarray
        Frequencies of the cell it solved, shape
        (n_wave_numbers, n_bands).

    Raises
    ------
    ValueError
        If the cell it picked is not buildable, or if the frequencies come
        back unusable.
    """
    source_folder = ROOT / "src" if source_folder is None else Path(source_folder)
    bounds = DESIGN_BOUNDS if bounds is None else bounds

    # The centre of the box, so the check exercises a cell the run could
    # actually propose rather than whatever the dataclass defaults hold
    design = bounds.mean(0)
    if not bool(is_feasible(design)):
        raise ValueError(
            f"The centre of the bounds, {design.tolist()}, is not buildable, "
            "so it cannot serve as a check; revise the bounds."
        )

    log.info("Checking the MATLAB solver on design %s", _format(design))
    started = time.perf_counter()

    with MatlabCaller(source_folder) as caller:
        caller.check()
        frequencies = matlab_evaluator(caller, settings)(design)

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


def run_single_objective(
    evaluate: Callable[[Tensor], np.ndarray],
    config: RunConfig = RunConfig(),
) -> RunHistory:
    """Drive one single-objective BO-CF run to its iteration budget.

    Parameters
    ----------
    evaluate : callable
        Maps a design of shape (4,) to frequencies of shape
        (n_wave_numbers, n_bands), as matlab_evaluator builds.
    config : RunConfig
        Bounds, budget and surrogate shape of the run.

    Returns
    -------
    RunHistory
        Every design solved, its relative gaps, and the incumbent trace.

    Raises
    ------
    ValueError
        If fewer than two of the initial designs solve, leaving nothing a
        surrogate could be fitted to.
    RuntimeError
        If max_failures proposals fail to solve in a row.
    """
    log.info(
        "Starting a single-objective run: %d initial designs, %d rounds of %d "
        "workers, %d solves in all, seed %d",
        config.n_initial, config.n_iterations, config.n_workers,
        config.n_initial + config.n_iterations * config.n_workers, config.seed,
    )
    log.info("Bounds lower %s", _format(config.bounds[0]))
    log.info("Bounds upper %s", _format(config.bounds[1]))

    designs = _initial_designs(config)
    solved: list[Tensor] = []
    measured: list[np.ndarray] = []
    started = time.perf_counter()

    log.info(
        "Solving %d initial designs across %d workers",
        len(designs), config.n_workers,
    )
    for position, (design, gaps) in enumerate(
        zip(designs, _solve_many(evaluate, designs, config.n_workers)), start=1,
    ):
        if gaps is None:
            continue

        solved.append(design)
        measured.append(gaps)
        log.info(
            "  initial %d of %d %s: relative gap %.6f over %d band pairs",
            position, len(designs), _format(design),
            maximum_relative_gap(gaps), len(gaps),
        )

    if len(solved) < 2:
        raise ValueError(
            f"Only {len(solved)} of the {config.n_initial} initial designs "
            "solved, which is too few to fit a surrogate; check the solver "
            "settings and the bounds."
        )

    log.info(
        "Solved %d of %d initial designs in %.1f s, incumbent %.6f",
        len(solved), config.n_initial, time.perf_counter() - started,
        max(maximum_relative_gap(row) for row in measured),
    )

    failures = 0
    for iteration in range(1, config.n_iterations + 1):
        started = time.perf_counter()
        design_tensor = torch.stack(solved)
        target_tensor = torch.as_tensor(np.stack(measured), dtype=torch.double)

        log.info(
            "Iteration %d of %d: fitting %d surrogates to %d designs",
            iteration, config.n_iterations, target_tensor.shape[1], len(solved),
        )
        model = fit_surrogates(build_surrogates(
            design_tensor, target_tensor, config.bounds, config.surrogate,
        ))

        log.info(
            "  maximizing qLogEI for %d designs over the buildable box",
            config.n_workers,
        )
        acquisition = single_objective_acquisition(
            model, target_tensor, seed=config.seed + iteration,
        )
        proposals = propose_batch(
            acquisition,
            config.bounds,
            config.n_workers,
            clearance=config.clearance,
            seed=config.seed + iteration * config.n_workers,
        )
        for proposal in proposals:
            log.info("  proposed %s", _format(proposal))

        results = _solve_many(evaluate, proposals, config.n_workers)
        if all(gaps is None for gaps in results):
            failures += 1
            log.warning(
                "  every proposal in the round failed, %d of %d rounds in a row",
                failures, config.max_failures,
            )
            if failures >= config.max_failures:
                raise RuntimeError(
                    f"{failures} rounds lost every proposal in a row at "
                    f"iteration {iteration}; the run cannot make progress."
                )
            continue

        failures = 0
        incumbent = max(maximum_relative_gap(row) for row in measured)
        for proposal, gaps in zip(proposals, results):
            if gaps is None:
                continue

            solved.append(proposal)
            measured.append(gaps)

            # Against the incumbent as it stands part way through the round,
            # so only the design that actually took the lead is marked
            widest = maximum_relative_gap(gaps)
            improved = widest > incumbent
            incumbent = max(incumbent, widest)
            log.info(
                "  %s: relative gap %.6f%s",
                _format(proposal), widest, " (new best)" if improved else "",
            )

        log.info(
            "  round solved %d of %d, incumbent %.6f, %.1f s",
            sum(gaps is not None for gaps in results), len(proposals),
            incumbent, time.perf_counter() - started,
        )

    relative_gaps = torch.as_tensor(np.stack(measured), dtype=torch.double)
    incumbents = np.maximum.accumulate(
        [maximum_relative_gap(row) for row in measured]
    )

    return RunHistory(
        designs=torch.stack(solved),
        relative_gaps=relative_gaps,
        incumbents=incumbents,
    )


def main() -> None:
    """Check the solver, run one single-objective optimization, save it.

    Pass --check to stop after the solver check, which costs one engine start
    and one solve rather than a whole run.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    settings = SolverSettings()
    config = RunConfig()

    # A run is hours of solves, so the setup is proved on one cell first
    check_matlab(settings, bounds=config.bounds)
    if "--check" in sys.argv[1:]:
        return

    started = time.perf_counter()
    with MatlabPool(ROOT / "src", config.n_workers) as pool:
        history = run_single_objective(matlab_evaluator(pool, settings), config)

    output = ROOT / "single_objective_history.npz"
    np.savez(
        output,
        designs=history.designs.numpy(),
        relative_gaps=history.relative_gaps.numpy(),
        incumbents=history.incumbents,
    )

    log.info("Run finished in %.1f min", (time.perf_counter() - started) / 60)
    log.info("Solved %d designs", len(history.designs))
    log.info("Best relative gap %.6f", history.best_relative_gap)
    log.info("Best design %s", _format(history.best_design))
    log.info("Wrote %s", output)


def _initial_designs(config: RunConfig) -> Tensor:
    """Draw the buildable Sobol designs a run starts from.

    Parameters
    ----------
    config : RunConfig
        Bounds, count and clearance the draw has to satisfy.

    Returns
    -------
    Tensor
        Buildable designs of shape (n_initial, 4).

    Raises
    ------
    ValueError
        If the box does not yield n_initial buildable designs.
    """
    # Oversample rather than solve the constraints: about a third of the draws
    # are unbuildable under the default bounds
    drawn = draw_sobol_samples(
        bounds=config.bounds, n=64 * config.n_initial, q=1, seed=config.seed,
    ).squeeze(1)
    buildable = drawn[is_feasible(drawn, clearance=config.clearance)]

    log.info(
        "Sampled %d Sobol designs, %d buildable (%.0f%%), keeping %d",
        len(drawn), len(buildable), 100 * len(buildable) / len(drawn),
        config.n_initial,
    )

    if len(buildable) < config.n_initial:
        raise ValueError(
            f"Only {len(buildable)} of {len(drawn)} draws are buildable, short "
            f"of the {config.n_initial} the run starts from; revise the bounds."
        )

    return buildable[:config.n_initial]


def _solve_many(
    evaluate: Callable[[Tensor], np.ndarray],
    designs: Tensor,
    n_workers: int,
) -> list[np.ndarray | None]:
    """Measure several designs at once, in the order they were given.

    Parameters
    ----------
    evaluate : callable
        Solver the run was handed, safe to call from several threads when it
        is backed by a MatlabPool.
    designs : Tensor
        Designs of shape (n_designs, 4).
    n_workers : int
        Solves to keep in flight.

    Returns
    -------
    list of numpy.ndarray or None
        One entry per design, None where the solve or measurement failed.
    """
    if n_workers == 1:
        return [_solve(evaluate, design) for design in designs]

    # The engine call blocks inside its C extension rather than on the GIL,
    # so threads are enough and the surrogate stays in one process
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(partial(_solve, evaluate), designs))


def _format(design: Tensor) -> str:
    """Render one design compactly enough to sit on a log line."""
    return "[" + ", ".join(f"{value:.4g}" for value in design) + "]"


def _solve(
    evaluate: Callable[[Tensor], np.ndarray],
    design: Tensor,
) -> np.ndarray | None:
    """Measure one design's relative gaps, or report that it would not solve.

    Parameters
    ----------
    evaluate : callable
        Solver the run was handed.
    design : Tensor
        One design of shape (4,).

    Returns
    -------
    numpy.ndarray or None
        Relative gap per band pair, or None if the solve or the measurement
        failed.
    """
    # A cell that will not mesh, or whose eigensolve returns nothing usable,
    # costs one evaluation rather than the whole run
    try:
        return relative_gap(evaluate(design))
    except Exception as error:
        log.warning("Design %s did not solve: %s", design.tolist(), error)

        return None


if __name__ == "__main__":
    main()
