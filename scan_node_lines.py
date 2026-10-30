"""Score edge-to-edge axis-aligned lines by distance to retained crossing nodes."""

import numpy as np


def line_scan(coordinates, extent, sigma):
    """Each node contributes exp(-distance^2/(2*sigma^2)); max contribution is one."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 1 or not len(coordinates) or not np.isfinite(coordinates).all():
        raise ValueError("Expected nonempty finite node coordinates")
    if not isinstance(extent, int) or extent < 1 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Require positive image extent and finite positive distance tolerance")
    if np.any((coordinates < 0) | (coordinates > extent-1)):
        raise ValueError("Node coordinate is outside the image")
    positions = np.arange(extent, dtype=float)
    scores = np.exp(-0.5 * ((positions[:,None] - coordinates[None,:]) / sigma)**2).sum(axis=1)
    return positions, scores
