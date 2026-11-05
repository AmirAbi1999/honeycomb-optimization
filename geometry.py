"""Geometry feasibility for the honeycomb unit cell.

A cell carries all five HoneycombGeometry fields, in field order. The two
constraints are the limit checks generate_geometry.m makes before it builds a
cell, kept signed so an unbuildable cell reports by how much it misses.
Everything is torch.

This module contains:
    - HoneycombGeometry
    - constraint_margins
    - is_feasible
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HoneycombGeometry:
    """One unit cell, in the parameters generate_geometry.m is handed.

    Every length is in millimetres, the model unit the MATLAB solver works in.

    Attributes
    ----------
    hex_angle : float
        Angle between two adjacent hexagon edges, in degrees.
    hex_side_length : float
        Hexagon side length, in millimetres.
    hex_thickness : float
        Hexagon frame thickness, in millimetres.
    face_sheet_thickness : float
        Face-sheet thickness, in millimetres.
    cell_width : float
        Unit-cell width along the in-plane y-direction, in millimetres.
    """

    hex_angle: float
    hex_side_length: float
    hex_thickness: float
    face_sheet_thickness: float
    cell_width: float

    def __post_init__(self) -> None:
        """Reject a cell generate_geometry.m would refuse to build.

        Raises
        ------
        ValueError
            If the cell is not feasible.
        """
        cell = torch.tensor(astuple(self), dtype=torch.double)

        if not bool(is_feasible(cell)):
            raise ValueError(
                f"{self} is not a cell generate_geometry.m can build; its "
                f"margins are {constraint_margins(cell).tolist()}."
            )


def constraint_margins(cells: Tensor) -> Tensor:
    """Evaluate both geometry constraints of every cell.

    Parameters
    ----------
    cells : Tensor
        One cell of shape (5,), or a batch of shape (..., 5), carrying the
        HoneycombGeometry fields in field order on the trailing axis.

    Returns
    -------
    Tensor
        The available gap and what is left of it once the walls move, stacked
        along a trailing axis, shape (..., 2). A cell is buildable where both
        are strictly positive.

    Raises
    ------
    ValueError
        If cells does not carry five values on its trailing axis.
    """
    if cells.ndim == 0 or cells.shape[-1] != 5:
        raise ValueError(
            "cells must carry the five HoneycombGeometry fields on the trailing "
            f"axis; received shape {tuple(cells.shape)}."
        )

    hex_angles = cells[..., 0]
    side_lengths = cells[..., 1]

    # generate_geometry.m measures the frame thickness normal to the wall and
    # works in its projection onto y, which is what its default converts to
    projected_thickness = cells[..., 2] / torch.sin(torch.deg2rad(hex_angles))
    available_gap = side_lengths - 2.0 * projected_thickness

    # tan(90 - angle) below 90 degrees is -tan(angle - 90), so one slant serves
    # both of the movement checks generate_geometry.m makes
    slants = torch.tan(torch.deg2rad(hex_angles - 90.0))
    total_movement = torch.where(
        hex_angles < 90.0,
        -2.0 * (side_lengths - projected_thickness) * slants,
        2.0 * projected_thickness * slants,
    )

    return torch.stack((available_gap, available_gap - total_movement), -1)


def is_feasible(cells: Tensor, *, clearance: float = 0.0) -> Tensor:
    """Return whether each cell is physically valid and clears both margins.

    Parameters
    ----------
    cells : Tensor
        One cell of shape (5,), or a batch of shape (..., 5).
    clearance : float
        Margin a cell has to clear, zero being the limit generate_geometry.m
        itself enforces. Raise it to leave room for a solver that only reaches
        a constraint to within its own tolerance.

    Returns
    -------
    Tensor
        Boolean of shape (...), True where the cell can be built.

    Raises
    ------
    ValueError
        If cells fails any check constraint_margins makes.
    """
    # face_sheet_thickness and cell_width reach neither margin, so a NaN,
    # infinite or negative one clears both and is still nothing
    # generate_geometry.m would accept; a NaN or infinite angle already fails
    # its own two comparisons
    lengths = cells[..., 1:]
    valid = (
        (cells[..., 0] > 0.0)
        & (cells[..., 0] < 180.0)
        & (lengths.isfinite() & (lengths > 0.0)).all(-1)
    )

    return valid & (constraint_margins(cells) > clearance).all(-1)
