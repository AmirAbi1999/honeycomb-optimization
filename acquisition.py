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

from geometry import constraint_margins


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
        stop comparing between them.
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
        to, or if reference_point does not hold exactly two finite values.
    """
    targets = _model_tensor(model, targets)
    _validate_targets(targets, model.num_outputs, paired=True)
    reference = _model_tensor(model, reference_point)

    if reference.shape != torch.Size([2]) or not reference.isfinite().all():
        raise ValueError(
            "reference_point must hold exactly two finite values, a bandwidth "
            f"and a negated mid-frequency; received {reference.tolist()}."
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
        Margin the returned cell keeps in hand. generate_geometry.m rejects a
        cell on the limit itself, and SLSQP is free to stop exactly there.
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
    buildable = drawn[(constraint_margins(drawn) > clearance).all(-1)]

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

    starts = buildable[promise.topk(num_restarts).indices]

    # Both constraints are strict in generate_geometry.m and SLSQP only
    # promises to reach zero, so each is offset by the clearance it keeps
    margins = [
        lambda design, index=index: constraint_margins(design)[index] - clearance
        for index in (0, 1)
    ]

    candidate, _ = optimize_acqf(
        acquisition,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        nonlinear_inequality_constraints=[(margin, True) for margin in margins],
        batch_initial_conditions=starts.unsqueeze(1),
        options={"maxiter": max_iterations, "ftol": tolerance},
    )
    candidate = candidate.squeeze(0)

    # SLSQP reports success on a constraint it only nearly satisfies, and a
    # cell the solver refuses is worth less than a duller one it will build
    if bool((constraint_margins(candidate) > clearance).all()):
        return candidate

    return starts[0]


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
