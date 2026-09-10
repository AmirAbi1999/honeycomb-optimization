"""Acquisition functions for Bayesian optimization of composite band gaps.

Both runs apply their outer function to posterior samples, never to the
posterior mean, which is what makes them composite. One design carries every
band pair, so the multi-objective objective fans it out into the front points
its pairs occupy and qLogEHVI weighs them together, at a cost of
2 ** (q * n_pairs) subsets that holds a proposal to one design at a time.

This module contains:
    - single_objective_acquisition
    - multi_objective_acquisition
    - propose_design
    - propose_batch
"""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from geometry import constraint_margins, is_feasible


def single_objective_acquisition(
    model: SingleTaskGP,
    targets: Tensor,
    *,
    mc_samples: int = 256,
    seed: int | None = None,
) -> qLogExpectedImprovement:
    """Build the acquisition that chases the widest relative gap.

    Parameters
    ----------
    model : botorch.models.SingleTaskGP
        Surrogates fitted to relative gaps, one output per band pair.
    targets : Tensor
        Observed relative gaps of shape (n_designs, n_pairs).
    mc_samples : int
        Quasi-random posterior samples the expectation is taken over.
    seed : int, optional
        Fixes the draw, so one optimization sees a single surface.

    Returns
    -------
    botorch.acquisition.logei.qLogExpectedImprovement
        Acquisition to maximize at q = 1.

    Raises
    ------
    ValueError
        If targets is not a finite matrix of the width the model was fitted to.
    """
    targets = _model_tensor(model, targets)
    _validate_targets(targets, model.num_outputs)

    # The incumbent is the outer function on observed data, never on the
    # posterior mean, which would quietly discard the composite structure
    return qLogExpectedImprovement(
        model=model,
        best_f=targets.max(),
        sampler=SobolQMCNormalSampler(torch.Size([mc_samples]), seed=seed),
        objective=GenericMCObjective(lambda samples, X=None: samples.amax(-1)),
    )


def multi_objective_acquisition(
    model: SingleTaskGP,
    targets: Tensor,
    reference_point: Tensor,
    *,
    mc_samples: int = 256,
    seed: int | None = None,
) -> qLogExpectedHypervolumeImprovement:
    """Build the acquisition that grows the width against mid-frequency front.

    Parameters
    ----------
    model : botorch.models.SingleTaskGP
        Surrogates fitted to interleaved bandwidths and mid-frequencies.
    targets : Tensor
        Observed targets of shape (n_designs, 2 * n_pairs). Every band pair of
        every design is a point the front is built from.
    reference_point : Tensor
        Two values, as (bandwidth, negated mid-frequency), bounding the
        dominated region. Held fixed across iterations, or the hypervolumes
        stop comparing between them. Its bandwidth decides whether a closed
        gap can reach the front, so it may not be negative.
    mc_samples : int
        Quasi-random posterior samples the expectation is taken over.
    seed : int, optional
        Fixes the draw, so one optimization sees a single surface.

    Returns
    -------
    qLogExpectedHypervolumeImprovement
        Acquisition to maximize at q = 1.

    Raises
    ------
    ValueError
        If targets is not a finite matrix of the width the model was fitted
        to, if reference_point does not hold exactly two finite values, or if
        its bandwidth is negative.
    """
    targets = _model_tensor(model, targets)
    _validate_targets(targets, model.num_outputs, paired=True)
    reference = _model_tensor(model, reference_point)

    if reference.shape != torch.Size([2]) or not reference.isfinite().all():
        raise ValueError(
            "reference_point must hold exactly two finite values, a bandwidth "
            f"and a negated mid-frequency; received {reference.tolist()}."
        )

    # Widths stay signed so the surrogate can see how far a pair is from
    # opening, and a negative reference would then pay hypervolume for pairs
    # that never open at all; zero keeps the front to real gaps
    if reference[0] < 0.0:
        raise ValueError(
            "reference_point's bandwidth may not be negative, or a band pair "
            "whose bands overlap earns hypervolume; received "
            f"{reference[0].item()}."
        )

    n_pairs = targets.shape[1] // 2

    return qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=reference,
        partitioning=FastNondominatedPartitioning(
            reference, Y=_front_points(targets, n_pairs).flatten(0, 1),
        ),
        sampler=SobolQMCNormalSampler(torch.Size([mc_samples]), seed=seed),
        objective=_BandPairObjective(n_pairs),
    )


def propose_design(
    acquisition: AcquisitionFunction,
    bounds: Tensor,
    *,
    clearance: float = 1e-6,
    num_restarts: int = 8,
    raw_samples: int = 256,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
    seed: int | None = None,
) -> Tensor:
    """Maximize one acquisition over the buildable part of the box.

    Parameters
    ----------
    acquisition : botorch.acquisition.AcquisitionFunction
        Either acquisition this module builds.
    bounds : Tensor
        Search bounds of shape (2, n_parameters), lower row first.
    clearance : float
        Margin the restarts keep in hand and SLSQP aims at, so that its own
        constraint tolerance still lands the answer above the limit itself.
    num_restarts, raw_samples : int
        Gradient restarts kept, and buildable candidates they are drawn from.
    max_iterations, tolerance : int, float
        SLSQP iteration cap and convergence tolerance.
    seed : int, optional
        Fixes the Sobol draw the restarts are chosen from.

    Returns
    -------
    Tensor
        One design of shape (n_parameters,), buildable by construction.

    Raises
    ------
    ValueError
        If a fanned acquisition carries pending points, or if fewer than
        num_restarts of the raw candidates are buildable.
    """
    # A pending point joins the q-batch, and the fan-out gives it n_pairs
    # front points of its own for the subset enumeration to double on
    if (
        isinstance(acquisition, qLogExpectedHypervolumeImprovement)
        and acquisition.X_pending is not None
    ):
        raise ValueError(
            f"acquisition carries {len(acquisition.X_pending)} pending points, "
            "which the band-pair fan-out makes exponentially expensive; "
            "propose one design at a time instead."
        )

    bounds = _model_tensor(acquisition.model, bounds)
    drawn = draw_sobol_samples(bounds=bounds, n=raw_samples, q=1, seed=seed)
    drawn = drawn.squeeze(1)
    buildable = drawn[is_feasible(drawn, clearance=clearance)]

    if len(buildable) < num_restarts:
        raise ValueError(
            f"Only {len(buildable)} of {raw_samples} candidates clear the "
            f"geometry constraints, short of the {num_restarts} restarts; "
            "increase raw_samples or revise the bounds."
        )

    # SLSQP cannot start outside the feasible set, so the restarts are the
    # most promising buildable draws rather than whatever BoTorch would pick
    with torch.no_grad():
        promise = acquisition(buildable.unsqueeze(1))

    best_promise, best_start = promise.max(0)
    starts = buildable[promise.topk(num_restarts).indices]

    # SLSQP drives an active constraint to its own zero, so each margin is
    # offset by the clearance the answer is required to keep
    margins = [
        lambda design, index=index: constraint_margins(design)[index] - clearance
        for index in (0, 1)
    ]

    candidate, value = optimize_acqf(
        acquisition,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        nonlinear_inequality_constraints=[(margin, True) for margin in margins],
        batch_initial_conditions=starts.unsqueeze(1),
        options={"maxiter": max_iterations, "ftol": tolerance},
    )
    candidate = candidate.squeeze(0)

    # The clearance is what SLSQP aims at, not what its answer has to hit: it
    # reaches a constraint only to within its own tolerance, and asking the
    # result to clear the buffer as well would spend it twice. Acceptance is
    # the limit itself, and a restart the solver left worse than it found it
    # loses to the draw it started from
    if bool(is_feasible(candidate)) and value > best_promise:
        return candidate

    return buildable[best_start]


def propose_batch(
    acquisition: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    *,
    clearance: float = 1e-6,
    num_restarts: int = 8,
    raw_samples: int = 256,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
    seed: int | None = None,
) -> Tensor:
    """Propose several designs, each one conditioned on those before it.

    Parameters
    ----------
    acquisition : botorch.acquisition.AcquisitionFunction
        Acquisition to maximize, which has to accept pending points.
    bounds : Tensor
        Search bounds of shape (2, n_parameters), lower row first.
    batch_size : int
        Designs to propose, normally one per solver worker.
    clearance, num_restarts, raw_samples, max_iterations, tolerance : float, int
        Passed through to propose_design for each design in turn.
    seed : int, optional
        Fixes the Sobol draws, offset by each design's position so they do
        not all start from the same restarts.

    Returns
    -------
    Tensor
        Designs of shape (batch_size, n_parameters), buildable by
        construction.

    Raises
    ------
    ValueError
        If batch_size is not positive, or if propose_design rejects the
        acquisition once it carries pending points.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive; received {batch_size}.")

    pending = acquisition.X_pending
    proposals: list[Tensor] = []

    # Greedy rather than a joint q-batch: qLogEI integrates over the posterior
    # at the pending points, so each design is chosen knowing the others are
    # already on their way to a solver, and the restarts stay a q = 1 problem
    # that the geometry constraints can still be handed to
    try:
        for position in range(batch_size):
            if proposals:
                acquisition.set_X_pending(torch.stack(proposals))

            proposals.append(propose_design(
                acquisition,
                bounds,
                clearance=clearance,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                max_iterations=max_iterations,
                tolerance=tolerance,
                seed=None if seed is None else seed + position,
            ))
    finally:
        # The caller's acquisition is left as it was handed over, pending
        # points included, however the loop ended
        acquisition.set_X_pending(pending)

    return torch.stack(proposals)


class _BandPairObjective(MCMultiOutputObjective):
    """Fan one design out into the front points its band pairs occupy.

    Parameters
    ----------
    n_pairs : int
        Band pairs each design carries, one front point apiece.
    """

    # One design leaves n_pairs points behind, so the objective returns more q
    # entries than X carries; _compute_log_qehvi reads its own q back off the
    # objective, and BoTorch stands the generic check down the same way for
    # its own set-valued risk measures
    _verify_output_shape = False

    def __init__(self, n_pairs: int) -> None:
        super().__init__()
        self.n_pairs = n_pairs

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        """Return the (..., q * n_pairs, 2) front points of every sample.

        Parameters
        ----------
        samples : Tensor
            Posterior draws of shape (..., q, 2 * n_pairs).
        X : Tensor, optional
            Unused, and only in the signature BoTorch calls through.

        Returns
        -------
        Tensor
            Front points of shape (..., q * n_pairs, 2).
        """
        return _front_points(samples, self.n_pairs).flatten(-3, -2)


def _model_tensor(model: SingleTaskGP, values: Tensor) -> Tensor:
    """Put values on the dtype and device the model was fitted on."""
    train_inputs = model.train_inputs[0]

    return torch.as_tensor(
        values, dtype=train_inputs.dtype, device=train_inputs.device,
    )


def _front_points(interleaved: Tensor, n_pairs: int) -> Tensor:
    """Split interleaved targets into two objectives that both maximize.

    Parameters
    ----------
    interleaved : Tensor
        Bandwidths and mid-frequencies alternating on the trailing axis,
        shape (..., 2 * n_pairs).
    n_pairs : int
        Band pairs the trailing axis carries.

    Returns
    -------
    Tensor
        Shape (..., n_pairs, 2), the mid-frequency negated so lower scores
        higher.
    """
    paired = interleaved.unflatten(-1, (n_pairs, 2))

    return torch.stack((paired[..., 0], -paired[..., 1]), dim=-1)


def _validate_targets(
    targets: Tensor,
    n_outputs: int,
    *,
    paired: bool = False,
) -> None:
    """Reject targets the surrogates cannot have been fitted to.

    Parameters
    ----------
    targets : Tensor
        Observed targets, as handed to either acquisition builder.
    n_outputs : int
        Outputs the model carries, one per target column.
    paired : bool
        Whether the columns interleave a bandwidth and a mid-frequency, and
        so have to come in twos.

    Raises
    ------
    ValueError
        If targets is not a finite matrix of the width the model was fitted to.
    """
    shape = tuple(targets.shape)

    if targets.ndim != 2 or targets.shape[0] < 1 or targets.shape[1] != n_outputs:
        raise ValueError(
            f"targets must be a matrix of one row per design and {n_outputs} "
            f"columns, the outputs the model carries; received shape {shape}."
        )

    if paired and n_outputs % 2 != 0:
        raise ValueError(
            "targets must interleave one bandwidth and one mid-frequency per "
            f"band pair, so its columns come in twos; received shape {shape}."
        )

    if not targets.isfinite().all():
        raise ValueError(f"targets must hold only finite values; got {shape}.")
