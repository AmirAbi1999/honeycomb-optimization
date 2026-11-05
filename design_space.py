"""The space the optimizer searches, and the honeycomb cells it maps onto.

A design holds the searched parameters only, in the order searched lists them;
a cell holds all five, in HoneycombGeometry's field order, with every fixed
parameter filled in. Surrogates and acquisitions only ever see designs, so a
fixed parameter never becomes a dimension they have to fit.

This module contains:
    - DesignSpace
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import torch
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from geometry import HoneycombGeometry, is_feasible


@dataclass(frozen=True)
class DesignSpace:
    """Which cell parameters the optimizer searches, and what the rest are held at.

    Attributes
    ----------
    searched : dict of str to tuple of float
        Searched parameters, each mapped to its (lower, upper) bound.
    fixed : dict of str to float
        Parameters held constant, each mapped to its value.
    """

    searched: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "hex_angle": (70.0, 165.0),
        "hex_side_length": (4.0, 10.0),
        "hex_thickness": (0.5, 2.0),
        "face_sheet_thickness": (0.4, 1.5),
    })
    fixed: dict[str, float] = field(default_factory=lambda: {"cell_width": 1.0})

    def __post_init__(self) -> None:
        """Reject a space that misses or repeats a cell parameter, or has no width.

        Raises
        ------
        ValueError
            If the cell parameters are not each searched or fixed exactly once,
            or if a lower bound does not sit below its upper.
        """
        cell_parameters = [parameter.name for parameter in fields(HoneycombGeometry)]
        named = sorted([*self.searched, *self.fixed])
        if named != sorted(cell_parameters):
            raise ValueError(
                f"Every cell parameter of {cell_parameters} must be searched or "
                f"fixed exactly once; received {named}."
            )

        narrow = [
            name for name, (lower, upper) in self.searched.items() if lower >= upper
        ]
        if narrow:
            raise ValueError(
                f"Every searched lower bound must sit below its upper; {narrow} do "
                "not, and a parameter held at one value belongs in fixed."
            )

    @property
    def bounds(self) -> Tensor:
        """Return the bounds the optimizer searches, shape (2, n_searched)."""
        return torch.tensor(list(self.searched.values()), dtype=torch.double).T

    def to_cells(self, designs: Tensor) -> Tensor:
        """Turn designs into cells, filling in every fixed parameter.

        Parameters
        ----------
        designs : Tensor
            Designs of shape (..., n_searched).

        Returns
        -------
        Tensor
            Cells of shape (..., 5), differentiable in the searched entries.
        """
        values = dict(zip(self.searched, designs.unbind(-1)))

        return torch.stack([
            values[parameter.name]
            if parameter.name in values
            else torch.full_like(designs[..., 0], self.fixed[parameter.name])
            for parameter in fields(HoneycombGeometry)
        ], dim=-1)

    def is_feasible(self, designs: Tensor, *, clearance: float = 0.0) -> Tensor:
        """Return whether each design's cell can be built.

        Parameters
        ----------
        designs : Tensor
            Designs of shape (..., n_searched).
        clearance : float
            Margin both constraints have to clear, as geometry.is_feasible
            takes it.

        Returns
        -------
        Tensor
            Boolean of shape (...).
        """
        return is_feasible(self.to_cells(designs), clearance=clearance)

    def sample(
        self,
        n_draws: int,
        *,
        seed: int | None = None,
        clearance: float = 0.0,
    ) -> Tensor:
        """Draw Sobol designs over the bounds and keep the buildable ones.

        Parameters
        ----------
        n_draws : int
            Sobol designs to draw before the unbuildable ones are dropped.
        seed : int, optional
            Fixes the draw.
        clearance : float
            Margin a kept design has to clear, as is_feasible takes it.

        Returns
        -------
        Tensor
            Buildable designs in draw order, shape (n_buildable, n_searched),
            with n_buildable at most n_draws.
        """
        drawn = draw_sobol_samples(
            bounds=self.bounds, n=n_draws, q=1, seed=seed,
        ).squeeze(1)

        return drawn[self.is_feasible(drawn, clearance=clearance)]
