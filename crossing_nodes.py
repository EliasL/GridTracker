"""Inspect positive crossing-score thresholds and place one centroid per component."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import center_of_mass, label, maximum
from scipy.signal import savgol_filter


def validate_scores(scores):
    if scores.ndim != 2 or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected a finite 2D crossing-score array in [0, 1]")


def threshold_coverage(scores):
    """Exact positive-threshold survival curve, normalized by the entire image."""
    validate_scores(scores)
    positive = np.sort(scores[scores > 0])
    if len(positive) < 2:
        raise ValueError("Need at least two positive score values for a logarithmic survival curve")
    thresholds = np.unique(positive)
    fractions = (len(positive) - np.searchsorted(positive, thresholds, side="right")) / scores.size
    nonzero = fractions > 0  # log y cannot display a zero retained fraction
    if not np.any(nonzero):
        raise ValueError("Need at least two distinct positive scores")
    return thresholds[nonzero], fractions[nonzero]


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


def filter_by_mass(nodes, sigma=1, mass_key="pixels", upper_limit=True):
    """One-pass lower mass cutoff with an optional upper cutoff, using population SD."""
    if not nodes or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Require nonempty regions and a positive finite sigma multiplier")
    sizes = np.array([node[mass_key] for node in nodes], dtype=float)
    mean, std = float(sizes.mean()), float(sizes.std(ddof=0))
    low = mean-sigma*std
    high = mean+sigma*std if upper_limit else None
    kept = [node for node in nodes if low <= node[mass_key] and (high is None or node[mass_key] <= high)]
    rejected = [node for node in nodes if not (low <= node[mass_key] and (high is None or node[mass_key] <= high))]
    return kept, rejected, {"mean": mean, "std": std, "sigma": sigma, "mass_key": mass_key,
                             "lower": low, "upper": high, "ddof": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("output/color_grayscale_comparison/crossing_scores.npy"))
    parser.add_argument("--model", type=Path, default=Path("output/color_grayscale_comparison/model.json"))
    parser.add_argument("--threshold", type=float, help="Manual override; default selects the start of the shallow-slope interval")
    parser.add_argument("--slope-tolerance", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("output/crossing_nodes_lower_mass"))
    parser.add_argument("--mass-sigma", type=float, default=1)
    parser.add_argument("--mass-measure", choices=["score", "pixels"], default="pixels")
    parser.add_argument("--upper-mass-limit", action="store_true", help="Also apply the upper mean+SD cutoff")
    args = parser.parse_args()
    scores = np.load(args.scores)
    thresholds, fractions = threshold_coverage(scores)
    selection = choose_threshold(scores, args.slope_tolerance) if args.threshold is None else None
    threshold = selection["threshold"] if selection else args.threshold
    mask, nodes = component_nodes(scores, threshold)
    unfiltered_fraction = float(mask.mean())
    nodes, rejected, mass_filter = filter_by_mass(nodes, args.mass_sigma, "score_mass" if args.mass_measure == "score" else "pixels", upper_limit=args.upper_mass_limit)
    labels, _ = label(mask, structure=np.ones((3,3), dtype=int))
    mask = np.isin(labels, [node["id"] for node in nodes])
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "threshold_coverage.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["positive_threshold", "fraction_of_entire_image_strictly_above_threshold"])
        writer.writerows(zip(thresholds, fractions))
    source = json.loads(args.model.read_text())["source"]
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    if image.size != scores.shape[::-1]:
        raise ValueError("Image dimensions differ from the crossing-score array")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(2 if selection else 1, 1, figsize=(10, 8 if selection else 5),
                             sharex=True, constrained_layout=True, squeeze=False)
    axis = axes[0,0]
    axis.step(thresholds, fractions, where="post", lw=1.5)
    axis.set_yscale("log")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100*y:g}%"))
    for t in (0.1, 0.2, 0.3):
        fraction = float(np.mean(scores > t))
        if fraction:
            axis.plot(t, fraction, "o")
            axis.annotate(f"t={t:g}: {fraction:.3%}", (t, fraction), xytext=(7, 10), textcoords="offset points")
    axis.axvline(threshold, color="black", ls="--", alpha=0.7, label=f"Node threshold {threshold:.3f}")
    if selection:
        x = np.array(selection["thresholds"])
        axis.plot(x, 10**np.array(selection["smoothed_log_fraction"]), color="purple", lw=1, label="Smoothed curve")
        axis.axvspan(threshold, selection["plateau_end"], color="green", alpha=0.1)
        minimum_t = selection["shallowest_threshold"]
        tangent_x = np.linspace(threshold, selection["plateau_end"], 100)
        tangent_y = np.interp(minimum_t, x, selection["smoothed_log_fraction"]) - selection["shallowest_decline"] * (tangent_x - minimum_t)
        axis.plot(tangent_x, 10**tangent_y, color="orange", lw=2, label="Tangent at shallowest slope")
        slope_axis = axes[1,0]
        slope_axis.plot(x, selection["decline"], label="Magnitude of smoothed log-slope")
        slope_axis.axhline(selection["shallowest_decline"], color="purple", ls="--", label=f"Shallowest rate: {selection['shallowest_decline']:.2f}")
        slope_axis.plot(selection["shallowest_threshold"], selection["shallowest_decline"], "o", color="purple")
        slope_axis.axhline(selection["decline_limit"], color="green", ls=":", label="20% above shallowest slope" if selection["tolerance"] == 0.2 else f"{selection['tolerance']:.0%} above shallowest slope")
        slope_axis.axvspan(threshold, selection["plateau_end"], color="green", alpha=0.1)
        slope_axis.axvline(threshold, color="black", ls="--")
        slope_axis.set(xlabel="Positive crossing-score threshold t", ylabel="-d log10(retained fraction) / dt",
                       title="Choose the first sustained shallow-slope interval", ylim=(0, min(20, max(selection["decline"]))))
        slope_axis.grid(alpha=0.25)
        slope_axis.legend()
        (args.output / "automatic_threshold.json").write_text(json.dumps(selection, indent=2) + "\n")
    axis.set(xlabel="Positive crossing-score threshold t", ylabel="Proportion of the entire image retained (log scale)",
             title="Retained pixels satisfy score > t; t > 0")
    axis.grid(which="both", alpha=0.25)
    axis.legend()
    fig.savefig(args.output / "threshold_coverage.png", dpi=180)
    fig.savefig(args.output / "threshold_coverage.pdf")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    axis.imshow(ImageEnhance.Brightness(image).enhance(3))
    heat = axis.imshow(scores, cmap="turbo", vmin=0, vmax=1, alpha=0.65 * mask)
    from matplotlib import patheffects
    markers = axis.scatter([n["x"] for n in nodes], [n["y"] for n in nodes],
                           s=40, marker="+", color="white", linewidths=1.2)
    markers.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="black")])
    fig.colorbar(plt.cm.ScalarMappable(norm=heat.norm, cmap=heat.cmap), ax=axis,
                 shrink=0.8, label="Crossing score (only above-threshold pixels shown)")
    axis.set_title(f"Crossing score > {threshold:.5f}: {len(nodes)} of {len(nodes)+len(rejected)} regions retained\nWhite crosses: centroids; {args.mass_measure} mass: lower mean - {args.mass_sigma:g} SD cutoff; upper cutoff {'on' if args.upper_mass_limit else 'off'}")
    axis.set_axis_off()
    fig.savefig(args.output / "nodes.png", dpi=180)
    plt.close(fig)
    Image.fromarray(np.uint8(mask) * 255).save(args.output / "mask.png")
    (args.output / "nodes.json").write_text(json.dumps({"threshold": threshold, "connectivity": 8,
        "retained_fraction": float(mask.mean()), "unfiltered_fraction": unfiltered_fraction,
        "mass_filter": mass_filter, "rejected_nodes": rejected, "source": source, "scores": str(args.scores.resolve()),
        "centroid": "uniform geometric pixel centroid", "nodes": nodes}, indent=2) + "\n")
    print(f"t={threshold:.6f}: {len(nodes)} retained, {len(rejected)} excluded; mass mean={mass_filter['mean']:.3f}, SD={mass_filter['std']:.3f}")


if __name__ == "__main__":
    main()
