"""Geometry feasibility for the honeycomb unit cell.

The two constraints are the limit checks generate_geometry.m makes before it
builds a cell, kept signed so an unbuildable cell reports by how much it
misses. Everything is torch.

This module contains:
    - HoneycombGeometry
    - constraint_margins
    - is_feasible
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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

    hex_angle: float = 135.0
    hex_side_length: float = 12.0
    hex_thickness: float = 1.0
    face_sheet_thickness: float = 1.0
    cell_width: float = 1.0

    def __post_init__(self) -> None:
        """Reject a cell generate_geometry.m would refuse to build.

        Raises
        ------
        ValueError
            If hex_angle is not strictly between 0 and 180 degrees, or if any
            length is not finite and positive.
        """
        if not 0.0 < self.hex_angle < 180.0:
            raise ValueError(
                "hex_angle must lie strictly between 0 and 180 degrees; "
                f"received {self.hex_angle}."
            )

        lengths = {
            "hex_side_length": self.hex_side_length,
            "hex_thickness": self.hex_thickness,
            "face_sheet_thickness": self.face_sheet_thickness,
            "cell_width": self.cell_width,
        }
        for name, value in lengths.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive; received {value}."
                )

    @property
    def margins(self) -> Tensor:
        """Return this cell's two constraints, positive where it is buildable."""
        return constraint_margins(torch.tensor([
            self.hex_angle, self.hex_side_length,
            self.hex_thickness, self.face_sheet_thickness,
        ], dtype=torch.double))

    @property
    def is_buildable(self) -> bool:
        """Return whether generate_geometry.m would accept this cell."""
        return bool((self.margins > 0.0).all())


def constraint_margins(parameters: Tensor) -> Tensor:
    """Evaluate both geometry constraints of every cell.

    Parameters
    ----------
    parameters : Tensor
        One cell of shape (4,), or a batch of shape (..., 4), carrying
        hex_angle, hex_side_length, hex_thickness and face_sheet_thickness in
        that order on the trailing axis.

    Returns
    -------
    Tensor
        The two constraints stacked along a trailing axis, shape (..., 2). A
        cell is buildable where both are strictly positive.

    Raises
    ------
    ValueError
        If parameters does not carry those four on its trailing axis.
    """
    if parameters.ndim == 0 or parameters.shape[-1] != 4:
        raise ValueError(
            "parameters must carry hex_angle, hex_side_length, hex_thickness "
            "and face_sheet_thickness on the trailing axis; received shape "
            f"{tuple(parameters.shape)}."
        )

    hex_angles = parameters[..., 0]
    side_lengths = parameters[..., 1]
    radians = torch.deg2rad(hex_angles)

    # generate_geometry.m measures the frame thickness normal to the wall and
    # works in its projection onto y, which is what its default converts to
    projected = parameters[..., 2] / torch.sin(radians)

    # tan(90 - angle) and tan(angle - 90) are the cotangent up to a sign, so
    # one cotangent serves both branches and zeroes itself at 90 degrees
    cotangent = torch.cos(radians) / torch.sin(radians)

    g1 = side_lengths - 2.0 * projected
    g2 = g1 - torch.where(
        hex_angles < 90.0,
        2.0 * (side_lengths - projected) * cotangent,
        -2.0 * projected * cotangent,
    )

    return torch.stack((g1, g2), -1)


def is_feasible(parameters: Tensor) -> Tensor:
    """Return whether each cell clears both geometry constraints.

    Parameters
    ----------
    parameters : Tensor
        One cell of shape (4,), or a batch of shape (..., 4).

    Returns
    -------
    Tensor
        Boolean of shape (...), True where the cell can be built.

    Raises
    ------
    ValueError
        If parameters fails any check constraint_margins makes.
    """
    return (constraint_margins(parameters) > 0.0).all(-1)
