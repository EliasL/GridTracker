"""Grid detection helpers and the folder-based application entry point."""
# Launch before importing numerical libraries, so worker thread limits apply.
if __name__ == '__main__':
    from run_temporal import main
    main()
    raise SystemExit

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, correlate1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks


def ridge_maps(rgb, width=2.0):
    """Color-neutral directional contrast, for light or dark strings."""
    if width <= 0 or not np.isfinite(width):
        raise ValueError("String width must be positive and finite")
    rgb = np.asarray(rgb, dtype=float) / 255
    signal = rgb @ np.array([0.2126, 0.7152, 0.0722])
    maps = []
    for axis in (0, 1):  # horizontal strings, vertical strings
        narrow = gaussian_filter1d(signal, width / 2, axis=axis)
        ridge = np.abs(narrow - gaussian_filter1d(signal, width * 3, axis=axis))
        scale = np.percentile(ridge, 99.5)
        if scale <= 1e-8:
            raise ValueError("No string contrast found; check image contrast")
        maps.append(np.clip(gaussian_filter(ridge / scale, 1), 0, 1))
    return maps


def color_evidence(rgb, maps, seed):
    """Learn string colors on seed segments and background colors inside the seed."""
    pixels = np.sqrt(np.asarray(rgb, dtype=float) / 255)
    strength = np.maximum(*maps)
    mask = Image.new("L", (pixels.shape[1], pixels.shape[0]))
    draw = ImageDraw.Draw(mask)
    for lines in (seed, seed.transpose(1, 0, 2)):
        for line in lines:
            draw.line([tuple(p) for p in line], fill=255, width=3)
    foreground = (np.asarray(mask) > 0) & (strength > 0.25)
    yy, xx = np.indices(strength.shape)
    region = ((xx >= seed[..., 0].min()) & (xx <= seed[..., 0].max())
              & (yy >= seed[..., 1].min()) & (yy <= seed[..., 1].max()))
    background = (strength < 0.1) & region
    if foreground.sum() < 30 or background.sum() < 30:
        raise ValueError("Not enough confident string/background pixels to learn colors")
    # Square-root RGB allocates more histogram resolution to the dark colors.
    # Background comparison rejects common colors shared with the ball or scene.
    densities = []
    for mask in (foreground, background):
        hist, _ = np.histogramdd(pixels[mask], bins=32, range=[(0, 1)] * 3)
        hist = gaussian_filter(hist, 0.8)
        densities.append(hist / hist.sum())
    coordinates = np.moveaxis(pixels * 32 - 0.5, -1, 0)
    positive, negative = [map_coordinates(hist, coordinates, order=1, mode="nearest")
                          for hist in densities]
    score = positive / (positive + negative + 1e-12)
    return gaussian_filter(score, 0.6)


def estimate_string_width(rgb, seed):
    """Gaussian FWHM from isolated perpendicular profiles, robustly pooled over both axes."""
    luminance = np.asarray(rgb, dtype=float) @ np.array([0.2126, 0.7152, 0.0722]) / 255
    offsets = np.arange(-12., 13.)
    measurements = []
    for axis, lines in enumerate((seed, seed.transpose(1, 0, 2))):
        for line in lines:
            for a, b in zip(line[:-1], line[1:]):
                tangent = (b - a) / np.linalg.norm(b - a)
                normal = np.array([-tangent[1], tangent[0]])
                for fraction in (0.3, 0.5, 0.7):
                    points = a + fraction * (b - a) + offsets[:, None] * normal
                    profile = sample(luminance, points)
                    slope, base = np.polyfit(offsets[abs(offsets) >= 9], profile[abs(offsets) >= 9], 1)
                    contrast = profile - base - slope * offsets
                    center = np.argmax(np.where(abs(offsets) <= 4, abs(contrast), -1))
                    amplitude = contrast[center]
                    if abs(amplitude) < 0.005:
                        continue
                    def residual(parameters):
                        level, tilt, height, position, sigma = parameters
                        return level + tilt * offsets + height * np.exp(-0.5 * ((offsets-position)/sigma)**2) - profile
                    fitted = least_squares(residual, [base, slope, amplitude, offsets[center], 2],
                        bounds=([-1,-1,-1,-6,0.4], [1,1,1,6,6]), max_nfev=100)
                    amplitude, position, sigma = fitted.x[2:]
                    error = np.sqrt(np.mean(fitted.fun**2)) / max(abs(amplitude), 1e-8)
                    if fitted.success and error < 0.2 and abs(position) < 4.5 and sigma < 5.5:
                        measurements.append({"direction": "horizontal" if axis == 0 else "vertical",
                            "width": float(2.35482 * sigma),
                            "point": (a + fraction * (b-a)).tolist(), "normal": normal.tolist(),
                            "parameters": fitted.x.tolist(), "profile": profile.tolist(),
                            "error": float(error)})
    if len(measurements) < 6:
        raise ValueError("Too few isolated string profiles to estimate width")
    return float(np.median([m["width"] for m in measurements])), measurements


def crossing_evidence(evidence, width):
    """Four-arm matched correlations, with negative support between the arms."""
    radius = max(6, int(np.ceil(3 * width)))
    gap = max(2, int(np.ceil(width)))
    def separated(row, column):
        row, column = row/row.sum(), column/column.sum()
        return correlate1d(correlate1d(evidence,row,axis=0,mode="nearest"),column,axis=1,mode="nearest")
    offsets = np.arange(-radius,radius+1)
    band = (abs(offsets)<=width/2).astype(float)
    positive = (offsets>=gap).astype(float)
    negative = (offsets<=-gap).astype(float)
    flank = (abs(offsets)>=gap).astype(float)
    responses = [separated(band,positive), separated(band,negative),
                 separated(positive,band), separated(negative,band)]
    background = separated(flank,flank)
    contrast = np.minimum.reduce(responses) - background
    score = np.clip(contrast / np.maximum(1 - background, 1e-6), 0, 1)
    return gaussian_filter(score, 0.6)


def sample(evidence, points):
    return map_coordinates(evidence, np.asarray(points)[..., ::-1].T, order=1, mode="nearest")


def segment(a, b):
    return np.linspace(a, b, max(5, int(np.ceil(np.linalg.norm(b - a) / 2))))


def estimate_pitch(maps):
    """Estimate pitch from the strongest regular 3x3 pattern; no whole-image fit."""
    height, width = maps[0].shape
    best = None
    for y in np.linspace(0, height // 2, 3, dtype=int):
        for x in np.linspace(0, width // 2, 3, dtype=int):
            triples = []
            for axis, evidence in enumerate(maps):
                patch = evidence[y:y + height // 2, x:x + width // 2]
                profile = gaussian_filter1d(patch.mean(axis=1 - axis), 1.5)
                peaks, _ = find_peaks(profile, prominence=max(0.01, profile.max() * 0.15), distance=8)
                candidates = []
                for i in range(len(peaks) - 2):
                    p = peaks[i:i + 3] + (y if axis == 0 else x)
                    gaps = np.diff(p)
                    if abs(gaps[0] - gaps[1]) < 0.15 * gaps.mean():
                        candidates.append(p)
                triples.append(candidates)
            for ys in triples[0]:
                for xs in triples[1]:
                    xx, yy = np.meshgrid(xs, ys)
                    seed = np.stack((xx, yy), axis=-1).astype(float)
                    supports = []
                    for axis, lines in enumerate((seed, seed.transpose(1, 0, 2))):
                        supports.append(np.mean([sample(maps[axis], segment(line[0], line[-1])).mean()
                                                 for line in lines]))
                    regularity = sum(np.std(np.diff(p)) / np.mean(np.diff(p)) for p in (ys, xs))
                    score = min(supports) / (1 + 5 * regularity)
                    if min(supports) > 0.15 and (best is None or score > best[0]):
                        best = score, seed
    if best is None:
        raise ValueError("No strong, regular 3x3 grid found in both string directions")
    return np.array([np.diff(best[1][..., 1], axis=0).mean(),
                     np.diff(best[1][..., 0], axis=1).mean()])


def mesh_uncertainty(seed, maps):
    """Area-integrated evidence deficit; density is independent of mesh area."""
    horizontal = np.array([[np.mean((1 - sample(maps[0], segment(a, b))) ** 2)
                            for a, b in zip(line[:-1], line[1:])] for line in seed])
    vertical = np.array([[np.mean((1 - sample(maps[1], segment(a, b))) ** 2)
                          for a, b in zip(line[:-1], line[1:])]
                         for line in seed.transpose(1, 0, 2)]).T
    xs, ys = seed[0, :, 0], seed[:, 0, 1]
    dx, dy = np.diff(xs), np.diff(ys)
    irregularity = 0.1 * (np.var(dx / np.median(dx)) + np.var(dy / np.median(dy)))
    cells = (horizontal[:-1] + horizontal[1:] + vertical[:, :-1] + vertical[:, 1:]) / 4
    areas = dy[:, None] * dx[None, :]
    total = float(np.sum(areas * (cells + irregularity)))
    return total, float(areas.sum()), total / float(areas.sum())


def find_seed(maps, max_uncertainty=0.45):
    """Expand all four corner candidates; stop acceptance at the first threshold failure."""
    pitch = estimate_pitch(maps)
    records, accepted = [], []
    for corner in ("top-left", "top-right", "bottom-left", "bottom-right"):
        axes = []
        for axis, evidence in enumerate(maps):
            length = evidence.shape[1 - axis]
            low = "left" in corner if axis == 0 else "top" in corner
            start = 0 if low else length - length // 3
            patch = (evidence[:, start:start + length // 3] if axis == 0 else
                     evidence[start:start + length // 3, :])
            profile = gaussian_filter1d(patch.mean(axis=1 - axis), 1.5)
            peaks, _ = find_peaks(np.pad(profile, 1), prominence=max(0.01, profile.max() * 0.12),
                                  distance=max(8, int(0.6 * pitch[axis])))
            axes.append(peaks - 1)
        stopped, largest = False, None
        for n in range(3, min(map(len, axes)) + 1):
            ys = axes[0][:n] if "top" in corner else axes[0][-n:]
            xs = axes[1][:n] if "left" in corner else axes[1][-n:]
            xx, yy = np.meshgrid(xs, ys)
            seed = np.stack((xx, yy), axis=-1).astype(float)
            total, area, density = mesh_uncertainty(seed, maps)
            stopped = stopped or density > max_uncertainty
            record = {"corner": corner, "n": n, "total": total, "area": area,
                      "density": density, "accepted": not stopped,
                      "xs": xs.tolist(), "ys": ys.tolist()}
            records.append(record)
            if not stopped:
                largest = record, seed
        if largest is not None:
            accepted.append(largest)
    if not accepted:
        raise ValueError("No corner has a 3x3 grid below the uncertainty threshold")
    record, seed = min(accepted, key=lambda item: (item[0]["density"], -item[0]["area"]))
    return seed, record, records


def plot_width_diagnostics(rgb, result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    measurements = result["width_samples"]
    if not measurements:
        return  # Explicit manual width overrides have no automatic measurements.
    width = result["width"]
    representative = min(measurements, key=lambda m: abs(m["width"] - width))
    point = np.array(representative["point"])
    normal = np.array(representative["normal"])
    parameters = representative["parameters"]
    center = point + parameters[3] * normal
    bright = np.asarray(ImageEnhance.Brightness(rgb).enhance(6))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    values = [m["width"] for m in measurements]
    bins = np.linspace(min(values)-0.2, max(values)+0.2, 16)
    for name, color in (("horizontal", "tab:blue"), ("vertical", "tab:orange")):
        data = [m["width"] for m in measurements if m["direction"] == name]
        axes[0,0].hist(data, bins=bins, alpha=0.6, label=f"{name}: n={len(data)}, median={np.median(data):.2f} px", color=color)
    axes[0,0].axvline(width, color="black", ls="--", label=f"Pooled median: {width:.2f} px")
    axes[0,0].set(xlabel="Apparent string width: Gaussian FWHM (original pixels)", ylabel="Accepted profiles", title="Width distribution from seed segments")
    axes[0,0].legend(fontsize=9)
    axes[0,1].imshow(bright)
    left = max(0, min(center[0]-150, rgb.width-300))
    top = max(0, min(center[1]-110, rgb.height-220))
    axes[0,1].set(xlim=(left, left+300), ylim=(top+220, top),
                  title="Grid zoom (6x brightness for display)")
    axes[0,1].add_patch(Rectangle(center-[35,20], 70, 40, fill=False, edgecolor="cyan", lw=1.5))
    axes[1,0].imshow(bright, interpolation="nearest")
    axes[1,0].set(xlim=(center[0]-35, center[0]+35), ylim=(center[1]+20, center[1]-20),
                  title=f"Median-width bar over actual {representative['direction']} string")
    for measured, color, label in ((width, "cyan", f"Pooled median {width:.2f} px"),
                                   (representative["width"], "yellow", f"This profile {representative['width']:.2f} px")):
        shift = np.array([-normal[1], normal[0]]) * (0 if color == "cyan" else 6)
        a, b = center - normal * measured/2 + shift, center + normal * measured/2 + shift
        axes[1,0].plot([a[0],b[0]], [a[1],b[1]], color=color, lw=2.5, marker="+", ms=10, label=label)
    axes[1,0].legend(fontsize=8, loc="lower right")
    for axis in (axes[0,1], axes[1,0]):
        axis.set_xlabel("Original image x (px)")
        axis.set_ylabel("Original image y (px)")
    offsets = np.arange(-12.,13.)
    fine = np.linspace(-12,12,400)
    base, tilt, height, position, sigma = parameters
    predicted = base + tilt*fine + height*np.exp(-0.5*((fine-position)/sigma)**2)
    axes[1,1].plot(offsets, representative["profile"], "o", ms=4, label="Original luminance samples")
    axes[1,1].plot(fine, predicted, label="Gaussian + linear background fit")
    for sign in (-1,1):
        axes[1,1].axvline(position + sign*representative["width"]/2, ls="--", color="tab:orange")
    axes[1,1].plot(fine, base+tilt*fine+height/2, ":", color="gray", label="Half contrast above/below background")
    axes[1,1].set(xlabel="Distance across string (original pixels)", ylabel="Original luminance (0-1)",
                  title=f"Representative fitted profile: FWHM {representative['width']:.2f} px")
    axes[1,1].legend(fontsize=8)
    fig.suptitle("Apparent image width, including blur; profiles avoid string crossings")
    fig.savefig(output / "width_diagnostics.png", dpi=180)
    fig.savefig(output / "width_diagnostics.pdf")
    plt.close(fig)
    import csv
    with (output / "width_measurements.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["direction", "fwhm_pixels", "profile_x", "profile_y", "relative_fit_error"])
        for measurement in measurements:
            writer.writerow([measurement["direction"], measurement["width"],
                             *measurement["point"], measurement["error"]])


def plot_uncertainty(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for corner in ("top-left", "top-right", "bottom-left", "bottom-right"):
        records = [r for r in result["candidates"] if r["corner"] == corner]
        for axis, field in zip(axes, ("total", "density")):
            line, = axis.plot([r["n"] for r in records], [r[field] for r in records], label=corner)
            for record in records:
                axis.plot(record["n"], record[field], "o", color=line.get_color(),
                          markerfacecolor=line.get_color() if record["accepted"] else "white")
    axes[0].set_ylabel("Total heuristic uncertainty (weighted pixel area)")
    axes[1].set_ylabel("Uncertainty per unit mesh area")
    axes[1].axhline(result["max_uncertainty"], color="black", ls="--", label="Acceptance threshold")
    selected = result["selected"]
    for axis, field in zip(axes, ("total", "density")):
        axis.scatter(selected["n"], selected[field], s=180, marker="*", color="black", zorder=5)
        axis.set_xlabel("Grid size n (n x n crossings)")
        axis.set_xticks(sorted({r["n"] for r in result["candidates"]}))
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Corner seed expansion: filled = accepted prefix; hollow = beyond first failure; star = selected")
    fig.savefig(output / "uncertainty.png", dpi=180)
    fig.savefig(output / "uncertainty.pdf")
    plt.close(fig)
    import csv
    with (output / "uncertainty.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["corner", "n", "total", "area", "density", "accepted"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in result["candidates"])

