"""Gaussian-process surrogates over the BO-CF inner function h.

One independent ARD Matern GP per target column, carried by a single batched
SingleTaskGP. Designs arrive in their own units and are normalized against
fixed bounds inside the model, which holds the scaling steady across BO
iterations. The solver is deterministic, so every GP interpolates its data.

This module contains:
    - SurrogateConfig
    - build_surrogates
    - fit_surrogates
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor


@dataclass(frozen=True)
class SurrogateConfig:
    """How the covariance of the BO-CF surrogates is put together.

    Attributes
    ----------
    smoothness : {0.5, 1.5, 2.5}
        Matern nu, lower for targets that kink where two modes cross.
    jitter : float
        Observation noise held fixed, as a fraction of each column's variance.
        A numerical floor for the Cholesky factor, not a noise model.
    """

    smoothness: float = 1.5
    jitter: float = 1e-6

    def __post_init__(self) -> None:
        """Reject a Matern order GPyTorch does not implement, or a bad jitter.

        Raises
        ------
        ValueError
            If smoothness is not one of 0.5, 1.5 or 2.5, or if jitter is not
            finite and positive.
        """
        if self.smoothness not in (0.5, 1.5, 2.5):
            raise ValueError(
                "smoothness must be one of 0.5, 1.5 or 2.5; received "
                f"{self.smoothness}."
            )

        if not 0.0 < self.jitter < float("inf"):
            raise ValueError(
                f"jitter must be finite and positive; received {self.jitter}."
            )


def build_surrogates(
    designs: Tensor,
    targets: Tensor,
    bounds: Tensor,
    config: SurrogateConfig = SurrogateConfig(),
) -> SingleTaskGP:
    """Build one independent GP per target column, unfitted.

    Parameters
    ----------
    designs : Tensor
        Evaluated designs of shape (n_designs, n_parameters), in the units
        bounds is given in.
    targets : Tensor
        Inner-function outputs of shape (n_designs, n_targets).
    bounds : Tensor
        Search bounds of shape (2, n_parameters), lower row first.
    config : SurrogateConfig
        Shape of the covariance each column is modelled with.

    Returns
    -------
    botorch.models.SingleTaskGP
        Batched model answering posterior with mean shape
        (n_designs, n_targets).

    Raises
    ------
    ValueError
        If the shapes disagree, if any value is not finite, or if a design
        falls outside bounds.
    """
    designs = torch.as_tensor(designs, dtype=torch.double)
    targets = torch.as_tensor(targets, dtype=torch.double)
    bounds = torch.as_tensor(bounds, dtype=torch.double)
    _validate_training_data(designs, targets, bounds)

    n_parameters = designs.shape[1]
    n_targets = targets.shape[1]

    # SingleTaskGP drops the batch dimension for a single output, and a kernel
    # batched anyway would hand the posterior a spurious leading axis
    batch_shape = torch.Size([n_targets] if n_targets > 1 else [])

    # FEM band gaps carry no noise, and an inferred noise level lets a GP
    # explain a kink where two modes cross as an outlier it can ignore.
    # Standardize divides the variance by the column's own, so the jitter is
    # scaled up front to land at config.jitter after it; a constant column is
    # left unscaled, as Standardize leaves it
    spread = targets.var(dim=0, keepdim=True)
    spread = spread.where(spread.sqrt() >= 1e-8, torch.ones_like(spread))

    return SingleTaskGP(
        designs,
        targets,
        train_Yvar=(config.jitter * spread).expand_as(targets),
        covar_module=_covariance_module(n_parameters, batch_shape, config),
        outcome_transform=Standardize(m=n_targets),
        # Fixed bounds, not the data range: a transform refitted each round
        # would move the length-scales out from under the previous fit
        input_transform=Normalize(d=n_parameters, bounds=bounds),
    )


def fit_surrogates(model: SingleTaskGP) -> SingleTaskGP:
    """Fit every independent GP by exact marginal likelihood.

    Parameters
    ----------
    model : botorch.models.SingleTaskGP
        Model returned by build_surrogates.

    Returns
    -------
    botorch.models.SingleTaskGP
        The same model, fitted in place.

    Raises
    ------
    botorch.exceptions.ModelFittingError
        If no optimizer restart converges.
    """
    marginal_log_likelihood = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(marginal_log_likelihood)

    return model


def _covariance_module(
    n_parameters: int,
    batch_shape: torch.Size,
    config: SurrogateConfig,
) -> ScaleKernel:
    """Build the kernel one target column is modelled with.

    Parameters
    ----------
    n_parameters : int
        Design parameters, one length-scale each under ARD.
    batch_shape : torch.Size
        Columns to hold independent hyperparameters for, empty for one column.
    config : SurrogateConfig
        Matern order the kernel is built with.

    Returns
    -------
    gpytorch.kernels.ScaleKernel
        Scaled ARD Matern kernel.
    """
    # The Gamma priors are what keep a fit on ten or twenty designs from
    # running the length-scales off to a degenerate optimum
    return ScaleKernel(
        MaternKernel(
            nu=config.smoothness,
            ard_num_dims=n_parameters,
            batch_shape=batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
        ),
        batch_shape=batch_shape,
        outputscale_prior=GammaPrior(2.0, 0.15),
    )


def _validate_training_data(
    designs: Tensor,
    targets: Tensor,
    bounds: Tensor,
) -> None:
    """Reject a training set no GP could be fitted to.

    Parameters
    ----------
    designs, targets, bounds : Tensor
        As handed to build_surrogates.

    Raises
    ------
    ValueError
        If the shapes disagree, if any value is not finite, or if a design
        falls outside bounds.
    """
    if designs.ndim != 2 or designs.shape[0] < 2:
        raise ValueError(
            "designs must be a (n_designs, n_parameters) matrix holding at "
            f"least two designs; received shape {tuple(designs.shape)}."
        )

    if (
        targets.ndim != 2
        or targets.shape[0] != designs.shape[0]
        or targets.shape[1] < 1
    ):
        raise ValueError(
            "targets must be a (n_designs, n_targets) matrix with one row per "
            f"design and at least one column; received {tuple(targets.shape)} "
            f"for {designs.shape[0]} designs."
        )

    if tuple(bounds.shape) != (2, designs.shape[1]):
        raise ValueError(
            f"bounds must have shape (2, {designs.shape[1]}) to match the "
            f"designs; received {tuple(bounds.shape)}."
        )

    for name, values in (
        ("designs", designs), ("targets", targets), ("bounds", bounds),
    ):
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must hold only finite values.")

    if (bounds[0] >= bounds[1]).any():
        raise ValueError(
            "Every lower bound must sit below its upper; received "
            f"{bounds.tolist()}."
        )

    outside = ((designs < bounds[0]) | (designs > bounds[1])).any(dim=1)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} of {designs.shape[0]} designs fall outside "
            "bounds, which would normalize them beyond [0, 1]."
        )
