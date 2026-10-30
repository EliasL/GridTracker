"""Compare the saved histogram input with width-profile color intensity grayscale."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import gaussian_filter

from gridtracker import color_evidence, crossing_evidence, ridge_maps, sample


def profile_color_grayscale(rgb, measurements):
    """Intensity along the learned RGB color, attenuated by chromaticity distance."""
    if len(measurements) < 6:
        raise ValueError("At least six accepted width profiles are required")
    pixels = np.asarray(rgb, dtype=float) / 255
    points = []
    for measurement in measurements:
        center = np.array(measurement["point"]) + measurement["parameters"][3] * np.array(measurement["normal"])
        points.extend(center + offset * measurement["width"] * np.array(measurement["normal"])
                      for offset in (-0.25, 0, 0.25))
    colors = np.column_stack([sample(pixels[..., channel], points) for channel in range(3)])
    valid = colors.sum(axis=1) > 1e-8
    excluded_black = int((~valid).sum())
    colors = colors[valid]
    if len(colors) < 6:
        raise ValueError('Fewer than six nonblack string-color samples remain')
    chromaticities = colors / colors.sum(axis=1, keepdims=True)
    target = np.median(chromaticities, axis=0)
    target /= target.sum()
    tolerance = max(0.03, float(np.quantile(np.linalg.norm(chromaticities - target, axis=1), 0.9)))
    direction = target / np.linalg.norm(target)
    reference = float(np.median(colors @ direction))
    if reference <= 1e-8:
        raise ValueError("String reference intensity is zero")
    model = {"target_rgb_chromaticity": target.tolist(), "chromaticity_tolerance": tolerance,
             "reference_intensity": reference, "profiles": len(measurements), "core_color_samples": len(colors),
             "excluded_black_samples": excluded_black,
             "median_core_rgb_255": np.median(colors * 255, axis=0).tolist()}
    return color_model_grayscale(rgb, model), model


def color_model_grayscale(rgb, model):
    """Apply a fixed learned palette, including across consecutive frames."""
    pixels = np.asarray(rgb, dtype=float)/255
    target = np.array(model['target_rgb_chromaticity'])
    tolerance, reference = model['chromaticity_tolerance'], model['reference_intensity']
    if target.shape != (3,) or not np.isfinite(target).all() or min(tolerance, reference) <= 0:
        raise ValueError('Invalid learned string-color model')
    direction = target/np.linalg.norm(target)
    chromaticity = pixels/np.maximum(pixels.sum(axis=2, keepdims=True), 1e-8)
    similarity = np.exp(-.5*np.sum((chromaticity-target)**2, axis=2)/tolerance**2)
    intensity = np.clip((pixels@direction)/reference, 0, 1)
    return gaussian_filter(similarity*intensity, .6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("output/confident_growth"))
    parser.add_argument("--output", type=Path, default=Path("output/color_grayscale_comparison"))
    args = parser.parse_args()
    previous = args.output / "previous"
    # Preserve originals before any new computation, and never replace a snapshot.
    previous.mkdir(parents=True, exist_ok=True)
    for name in ("crossing_overlay.png", "crossing_overlay.pdf", "crossing_scores.npy",
                 "color_evidence.png", "growth.json"):
        destination = previous / name
        if destination.exists():
            if destination.read_bytes() != (args.baseline / name).read_bytes():
                raise ValueError(f"Snapshot differs from current baseline: {destination}")
        else:
            shutil.copy2(args.baseline / name, destination)
    metadata = json.loads((previous / "growth.json").read_text())
    with Image.open(args.image) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
    n = metadata["selected"]["n"]
    seed = np.empty((n, n, 2))
    for node in metadata["stages"][0]["nodes"]:
        seed[node["row"], node["column"]] = node["xy"]
    original = color_evidence(rgb, ridge_maps(rgb), seed)
    old_score = np.load(previous / "crossing_scores.npy")
    width = metadata["width_fwhm"]
    np.testing.assert_allclose(crossing_evidence(original, width), old_score, atol=1e-12)
    grayscale, model = profile_color_grayscale(rgb, metadata["width_samples"])
    new_score = crossing_evidence(grayscale, width)
    np.save(args.output / "grayscale.npy", grayscale)
    np.save(args.output / "crossing_scores.npy", new_score)
    Image.fromarray(np.uint8(grayscale * 255)).save(args.output / "grayscale.png")
    model.update(width_fwhm=width, window_size=2 * max(6, int(np.ceil(3 * width))) + 1,
                 source=str(args.image.resolve()), formula="grayscale = learned-color intensity x chromaticity similarity")
    (args.output / "model.json").write_text(json.dumps(model, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    background = ImageEnhance.Brightness(rgb).enhance(3)
    for column, (label, evidence, score) in enumerate((("Previous: string/background color histogram", original, old_score),
                                                     ("New: width-profile color intensity grayscale", grayscale, new_score))):
        gray_image = axes[0,column].imshow(evidence, cmap="gray", vmin=0, vmax=1)
        axes[0,column].set_title(label)
        axes[1,column].imshow(background)
        heat = axes[1,column].imshow(score, cmap="turbo", vmin=0, vmax=1, alpha=0.85*score)
        axes[1,column].set_title("Crossing-score overlay")
        for row in (0,1):
            axes[row,column].set_axis_off()
    fig.colorbar(gray_image, ax=axes[0,:], shrink=0.75, label="Scalar filter input (0-1)")
    fig.colorbar(plt.cm.ScalarMappable(norm=heat.norm, cmap=heat.cmap), ax=axes[1,:],
                 shrink=0.75, label="Crossing score (0-1, heuristic)")
    fig.suptitle(f"Same {model['window_size']}x{model['window_size']} filter, width {width:.2f} px; same scales and opacity = 0.85 x score")
    fig.savefig(args.output / "comparison.png", dpi=180)
    fig.savefig(args.output / "comparison.pdf")
    plt.close(fig)
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    main()
