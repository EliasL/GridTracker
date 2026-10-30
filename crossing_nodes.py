"""Select a crossing threshold, component centroids, and the lower pixel-mass cutoff."""

import numpy as np
from scipy.ndimage import center_of_mass, label, maximum
from scipy.signal import savgol_filter


def validate_scores(scores):
    if scores.ndim != 2 or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected a finite 2D crossing-score array in [0, 1]")




def choose_threshold(scores, tolerance=0.2, smoothing=0.08, allow_unsustained=False):
    """First sustained near-minimum decline rate of log10 retained image fraction."""
    validate_scores(scores)
    if not 0 < tolerance < 1 or not 0.02 <= smoothing <= 0.2:
        raise ValueError("Require 0 < slope tolerance < 1 and 0.02 <= smoothing <= 0.2")
    positive = np.sort(scores[scores > 0])
    pixel_floor = max(20, int(np.ceil(scores.size * 1e-4)))
    if len(positive) <= pixel_floor:
        raise ValueError("Too few positive pixels for reliable slope estimation")
    low, high = float(positive[0]), float(np.nextafter(positive[-pixel_floor], 0))
    if high <= low:
        raise ValueError("Too little positive-score variation for slope estimation")
    thresholds = np.linspace(low, high, 512)
    fractions = (len(positive) - np.searchsorted(positive, thresholds, side="right")) / scores.size
    window = int(smoothing * len(thresholds)) // 2 * 2 + 1
    log_fraction = np.log10(fractions)
    smoothed = savgol_filter(log_fraction, window, 1)
    decline = -savgol_filter(log_fraction, window, 1, deriv=1, delta=thresholds[1] - thresholds[0])
    indices = np.arange(len(thresholds))
    usable = (indices >= window//2) & (indices < len(indices)-window//2) & (decline > 0)
    if not usable.any():
        raise ValueError("No decreasing slope interval found")
    shallowest = float(decline[usable].min())
    band = usable & (decline <= (1+tolerance) * shallowest)
    indices = np.flatnonzero(band)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    runs = [run for run in runs if len(run) and
            thresholds[run[-1]] - thresholds[run[0]] >= 0.05 * (high-low)]
    sustained = bool(runs)
    if not sustained and not allow_unsustained:
        raise ValueError("No sustained shallow-slope interval found; inspect the curve")
    run = runs[0] if sustained else np.array([np.flatnonzero(usable)[np.argmin(decline[usable])]])
    return {"threshold": float(thresholds[run[0]]), "plateau_end": float(thresholds[run[-1]]),
            "tolerance": tolerance, "smoothing_fraction": smoothing, "pixel_floor": pixel_floor,
            "sustained": sustained,
            "shallowest_decline": shallowest,
            "shallowest_threshold": float(thresholds[np.flatnonzero(usable)[np.argmin(decline[usable])]]),
            "decline_limit": (1+tolerance)*shallowest,
            "thresholds": thresholds.tolist(), "smoothed_log_fraction": smoothed.tolist(),
            "decline": decline.tolist()}


def component_nodes(scores, threshold):
    validate_scores(scores)
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be strictly between zero and one")
    mask = scores > threshold
    labels, count = label(mask, structure=np.ones((3, 3), dtype=int))
    if not count:
        return mask, []
    ids = np.arange(1, count + 1)
    centers = center_of_mass(mask, labels, ids)  # geometric, not score-weighted
    sizes = np.bincount(labels.ravel())
    masses = np.bincount(labels.ravel(), weights=scores.ravel())
    peaks = maximum(scores, labels, ids)
    nodes = [{"id": int(i), "x": float(x), "y": float(y), "pixels": int(sizes[i]), "score_mass": float(masses[i]), "peak": float(peak)}
             for i, (y, x), peak in zip(ids, centers, peaks)]
    return mask, nodes


def filter_by_mass(nodes):
    """One-pass mean-minus-population-SD pixel cutoff, with no upper limit."""
    if not nodes:
        raise ValueError("Require nonempty crossing regions")
    sizes = np.array([node['pixels'] for node in nodes], dtype=float)
    mean, std = float(sizes.mean()), float(sizes.std(ddof=0))
    low = mean-std
    kept = [node for node in nodes if node['pixels'] >= low]
    rejected = [node for node in nodes if node['pixels'] < low]
    return kept, rejected, {'mean': mean, 'std': std, 'lower': low}
