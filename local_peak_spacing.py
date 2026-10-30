"""Mean spacing of all scan maxima inside a sliding quarter-image window."""
import argparse
import json
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=Path("output/node_line_scans/vertical_scan.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/local_peak_spacing"))
    args = parser.parse_args()
    scan = np.loadtxt(args.scan, delimiter=",", skiprows=1)
    np.testing.assert_array_equal(scan[:,0], np.arange(len(scan)))
    x, signal = scan[:,0], scan[:,1]
    peaks, width, centers, means, counts = sliding_spacing(signal)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output/"spacing.csv", np.column_stack((x,centers,means,counts)), delimiter=",",
               header="x,full_window_center,mean_peak_spacing,peaks_in_window", comments="")
    metadata = {"source":str(args.scan.resolve()),"window_width_pixels":width,"maxima_x":peaks.tolist(),
                "mean_spacing_left_edge":float(means[0]),"mean_spacing_right_edge":float(means[-1]),
                "statistic":"mean consecutive-maxima spacing; both maxima inside the full window",
                "boundary_rule":"clamp window center to nearest full-window center"}
    (args.output/"spacing.json").write_text(json.dumps(metadata,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex":False,"font.family":"DejaVu Sans"})
    fig, axes = plt.subplots(2,1,figsize=(12,7),sharex=True,constrained_layout=True)
    axes[0].plot(x,signal,label="Vertical-line x-scan")
    axes[0].plot(x[peaks],signal[peaks],"o",color="tab:red",ms=5,label=f"All {len(peaks)} local maxima")
    axes[0].set(ylabel="Node proximity score",title="Local maxima of the scan across x")
    axes[0].legend()
    axes[1].plot(x,means,color="tab:blue",label=f"Mean spacing inside {width:g}-pixel window")
    axes[1].set(xlabel="Scan position x (pixels)",ylabel="Mean spacing between maxima (pixels)",
                title="Sliding window = 1/4 image width; nearest full window used at the edges")
    for axis in axes:
        axis.axvspan(0,width/2,color="gray",alpha=.12)
        axis.axvspan(len(signal)-width/2,len(signal)-1,color="gray",alpha=.12)
        axis.axvline(width/2,color="gray",ls=":")
        axis.axvline(len(signal)-width/2,color="gray",ls=":")
        axis.grid(alpha=.25)
        axis.set_xlim(0,len(signal)-1)
    axes[1].legend()
    fig.savefig(args.output/"local_spacing.png",dpi=180)
    fig.savefig(args.output/"local_spacing.pdf")
    plt.close(fig)
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__":
    main()
