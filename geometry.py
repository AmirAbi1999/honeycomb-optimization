"""Geometry feasibility for the honeycomb unit cell.

The two constraints are the limit checks generate_geometry.m makes before it
builds a cell, kept signed so an unbuildable cell reports by how much it
misses rather than only that it does.

This module contains:
    - HoneycombGeometry
    - constraint_margins
    - is_feasible
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HoneycombGeometry:
    """One unit cell, in the parameters generate_geometry.m is handed.

    Attributes
    ----------
    hex_angle : float
        Angle between two adjacent hexagon edges, in degrees.
    hex_side_length : float
        Hexagon side length, in metres.
    hex_thickness : float
        Hexagon frame thickness, in metres.
    face_sheet_thickness : float
        Face-sheet thickness, in metres.
    cell_width : float
        Unit-cell width along the in-plane y-direction, in metres.
    """

    hex_angle: float = 135.0
    hex_side_length: float = 0.012
    hex_thickness: float = 0.001
    face_sheet_thickness: float = 0.001
    cell_width: float = 0.001

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
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive; received {value}."
                )

    @property
    def margins(self) -> np.ndarray:
        """Return this cell's two constraints, positive where it is buildable."""
        return constraint_margins([
            self.hex_angle, self.hex_side_length,
            self.hex_thickness, self.face_sheet_thickness,
        ])

    @property
    def is_buildable(self) -> bool:
        """Return whether generate_geometry.m would accept this cell."""
        return bool(np.all(self.margins > 0.0))


def constraint_margins(parameters: np.ndarray) -> np.ndarray:
    """Evaluate both geometry constraints of every cell.

    Parameters
    ----------
    parameters : numpy.ndarray
        One cell of shape (4,), or a batch of shape (..., 4), carrying
        hex_angle, hex_side_length, hex_thickness and face_sheet_thickness
        in that order on the trailing axis.

    Returns
    -------
    numpy.ndarray
        The two constraints stacked along a trailing axis, shape (..., 2).
        A cell is buildable where both are positive.

    Raises
    ------
    ValueError
        If parameters does not carry those four on its trailing axis.
    """
    parameters = np.asarray(parameters, dtype=np.float64)

    if parameters.ndim == 0 or parameters.shape[-1] != 4:
        raise ValueError(
            "parameters must carry hex_angle, hex_side_length, hex_thickness "
            "and face_sheet_thickness on the trailing axis; received shape "
            f"{parameters.shape}."
        )

    hex_angles = parameters[..., 0]
    side_lengths = parameters[..., 1]
    radians = np.deg2rad(hex_angles)

    # generate_geometry.m measures the frame thickness normal to the wall and
    # works in its projection onto y, which is what its default converts to
    projected = parameters[..., 2] / np.sin(radians)

    # tan(90 - angle) and tan(angle - 90) are the cotangent up to a sign, so
    # one cotangent serves both branches and zeroes itself at 90 degrees
    cotangent = np.cos(radians) / np.sin(radians)

    g1 = side_lengths - 2.0 * projected
    g2 = g1 - np.where(
        hex_angles < 90.0,
        2.0 * (side_lengths - projected) * cotangent,
        -2.0 * projected * cotangent,
    )

    return np.stack((g1, g2), axis=-1)


def is_feasible(parameters: np.ndarray) -> np.ndarray:
    """Return whether each cell clears both geometry constraints.

    Parameters
    ----------
    parameters : numpy.ndarray
        One cell of shape (4,), or a batch of shape (..., 4).

    Returns
    -------
    numpy.ndarray
        Boolean of shape (...), True where the cell can be built.

    Raises
    ------
    ValueError
        If parameters fails any check constraint_margins makes.
    """
    return np.all(constraint_margins(parameters) > 0.0, axis=-1)
