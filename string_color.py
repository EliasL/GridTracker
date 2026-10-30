"""Learn string chromaticity from width profiles and apply a fixed color model."""

import numpy as np
from scipy.ndimage import gaussian_filter

from detection import sample


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
