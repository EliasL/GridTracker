"""Mean spacing of all scan maxima inside a sliding quarter-image window."""

import numpy as np
from scipy.signal import find_peaks


def sliding_spacing(signal, window_fraction=0.25, peaks=None, allow_missing=False):
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all():
        raise ValueError("Require a nonempty finite one-dimensional scan")
    if not 0 < window_fraction <= 1:
        raise ValueError("Window fraction must be in (0, 1]")
    if peaks is None:
        peaks, _ = find_peaks(signal)  # All local maxima, with no prominence/distance filtering.
    else:
        peaks = np.asarray(peaks, dtype=int)
        if len(peaks) < 2 or np.any(np.diff(peaks) <= 0) or peaks[0] < 0 or peaks[-1] >= len(signal):
            raise ValueError("Require sorted unique in-range peak positions")
    width = window_fraction * len(signal)
    centers = np.clip(np.arange(len(signal)), width/2, len(signal)-width/2)
    means, counts = [], []
    for center in centers:
        local = peaks[(peaks >= center-width/2) & (peaks < center+width/2)]
        if len(local) < 2:
            if allow_missing:
                means.append(float("nan"))
                counts.append(len(local))
                continue
            raise ValueError(f"Fewer than two maxima in full window centered at {center:g}")
        means.append(float(np.mean(np.diff(local))))
        counts.append(len(local))
    return peaks, width, centers, np.array(means), np.array(counts)


def prune_split_peaks(peaks):
    """Retry all triples after each deletion, updating the median until a fixed point."""
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 2 or np.any(np.diff(peaks) <= 0):
        raise ValueError("Need at least two sorted unique peaks")
    reference = float(np.median(np.diff(peaks)))
    kept, removals = peaks.tolist(), []
    while len(kept) >= 3:
        gaps = np.diff(kept)
        reference = float(np.median(gaps))
        gains = []
        for i, (left, right) in enumerate(zip(gaps[:-1], gaps[1:]), 1):
            before = abs((left + right)/2 - reference)
            after = abs(left + right - reference)
            if left < reference and right < reference and after < before:
                gains.append((before-after, i, before, after))
        if not gains:
            break
        gain, i, before, after = max(gains, key=lambda item: item[0])
        removals.append({"peak":kept[i], "left":kept[i-1], "right":kept[i+1],
                         "before_error":float(before), "after_error":float(after), "reference_median":reference})
        del kept[i]
    return np.array(kept), reference, removals
