"""Fourier spacing, translation-only grid alignment, and close-point competition."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from crossing_nodes import filter_by_mass


def fourier_spacing(coordinates, extent, min_spacing=8):
    """Direct Fourier transform of unit point impulses, excluding DC and long trends."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 1 or not np.isfinite(coordinates).all() or not 0 < min_spacing < extent/3:
        raise ValueError("Require finite 1D coordinates and 0 < minimum spacing < extent/3")
    if len(coordinates) < 3:
        raise ValueError("Need at least three points and room for three grid periods")
    frequencies = np.linspace(3/extent, 1/min_spacing, 12000)
    coherence = abs(np.exp(-2j*np.pi*frequencies[:,None]*coordinates).mean(axis=1))
    peaks, _ = find_peaks(coherence)
    if not len(peaks):
        raise ValueError("No interior Fourier peak found in spacing search range")
    best = int(peaks[np.argmax(coherence[peaks])])
    result = minimize_scalar(lambda f: -abs(np.exp(-2j*np.pi*f*coordinates).mean()),
                             bounds=(frequencies[best-1], frequencies[best+1]), method="bounded",
                             options={"xatol": 1e-12})
    if not result.success:
        raise RuntimeError("Fourier peak refinement failed")
    alternatives = sorted(peaks, key=lambda i: -coherence[i])[:5]
    return 1/result.x, {"frequencies": frequencies, "coherence": coherence,
                        "peaks": [{"spacing": float(1/frequencies[i]), "coherence": float(coherence[i])}
                                  for i in alternatives]}


def wrapped_residual(coordinates, spacing, offset):
    return (coordinates-offset+spacing/2) % spacing - spacing/2


def fit_offset(coordinates, spacing):
    """Globally scan one period, then refine mean squared distance to grid lines."""
    coordinates = np.asarray(coordinates, dtype=float)
    candidates = np.linspace(0, spacing, 1024, endpoint=False)
    loss = np.mean(wrapped_residual(np.asarray(coordinates)[:,None], spacing, candidates)**2, axis=0)
    best = candidates[np.argmin(loss)]
    step = spacing / len(candidates)
    fitted = minimize_scalar(lambda offset: np.mean(wrapped_residual(coordinates, spacing, offset)**2),
                             bounds=(best-step, best+step), method="bounded")
    if not fitted.success:
        raise RuntimeError("Grid translation fit failed")
    return float(fitted.x % spacing)


def remove_close_points(points, spacing, offset, min_separation=0.5):
    """Keep grid-nearest points first; reject only competitors near a kept point."""
    if not 0 < min_separation < 1:
        raise ValueError("Minimum separation must be between zero and one grid spacing")
    residual = wrapped_residual(points, spacing, offset)
    distances = np.linalg.norm(residual, axis=1)
    pairs = sorted(cKDTree(points/spacing).query_pairs(min_separation))
    crowded = {index for pair in pairs for index in pair}
    neighbors = {index: set() for index in range(len(points))}
    for a, b in pairs:
        neighbors[a].add(b)
        neighbors[b].add(a)
    kept, removed_by = set(), {}
    for index in np.argsort(distances, kind="stable"):
        winners = kept & neighbors[index]
        if winners:
            removed_by[int(index)] = int(min(winners, key=lambda i: distances[i]))
        else:
            kept.add(int(index))
    return distances, pairs, crowded, kept, removed_by


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=Path("output/crossing_nodes_auto/nodes.json"))
    parser.add_argument("--output", type=Path, default=Path("output/fourier_point_grid"))
    parser.add_argument("--min-spacing", type=float, default=8)
    parser.add_argument("--min-separation", type=float, default=0.5)
    args = parser.parse_args()
    data = json.loads(args.nodes.read_text())
    all_nodes = sorted(data["nodes"] + data.get("rejected_nodes", []), key=lambda node: node["id"])
    nodes, small, mass_stats = filter_by_mass(all_nodes, mass_key="pixels", upper_limit=False)
    points = np.array([[node["x"],node["y"]] for node in nodes])
    with Image.open(data["source"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    spacing, spectra = [], []
    for axis, extent in enumerate(image.size):
        pitch, spectrum = fourier_spacing(points[:,axis], extent, args.min_spacing)
        spacing.append(pitch)
        spectra.append(spectrum)
    spacing = np.array(spacing)
    offset = np.array([fit_offset(points[:,axis], spacing[axis]) for axis in (0,1)])
    distances, pairs, crowded, kept, removed_by = remove_close_points(points, spacing, offset, args.min_separation)
    args.output.mkdir(parents=True, exist_ok=True)
    records = [dict(node, grid_distance_pixels=float(distances[i]), too_close=i in crowded,
                    keep=i in kept, removed_by_id=None if i in kept else nodes[removed_by[i]]["id"])
               for i, node in enumerate(nodes)]
    result = {"source": data["source"], "mass_filter": mass_stats, "mass_rejected": small,
              "spacing_xy": spacing.tolist(), "offset_xy": offset.tolist(), "rotation_degrees": 0,
              "min_separation_in_grid_units": args.min_separation,
              "close_pairs": [[nodes[a]["id"],nodes[b]["id"]] for a,b in pairs],
              "fourier_peaks": {name: spectrum["peaks"] for name,spectrum in zip(("x","y"),spectra)},
              "rms_grid_distance_pixels": float(np.sqrt(np.mean(distances**2))), "nodes": records}
    (args.output/"grid.json").write_text(json.dumps(result, indent=2)+"\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex":False, "font.family":"DejaVu Sans"})
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for axis, spectrum, pitch, name in zip(axes,spectra,spacing,("Horizontal (x)","Vertical (y)")):
        periods=1/spectrum["frequencies"]
        axis.plot(periods,spectrum["coherence"])
        axis.axvline(pitch,color="red",ls="--",label=f"Selected {pitch:.2f} px")
        axis.set(xlabel="Candidate grid spacing (pixels)",ylabel="Fourier coherence (0-1)",title=name,xlim=(8,210))
        axis.grid(alpha=.2);axis.legend()
    fig.savefig(args.output/"fourier_spectra.png",dpi=180);fig.savefig(args.output/"fourier_spectra.pdf")
    plt.close(fig)

    fig, axes=plt.subplots(2,1,figsize=(14,14),constrained_layout=True)
    background=ImageEnhance.Brightness(image).enhance(3)
    for axis in axes:
        axis.imshow(background)
        for x in np.arange(offset[0],image.width,spacing[0]): axis.axvline(x,color="cyan",lw=.7,alpha=.7)
        for y in np.arange(offset[1],image.height,spacing[1]): axis.axhline(y,color="cyan",lw=.7,alpha=.7)
        axis.set(xlim=(0,image.width-1),ylim=(image.height-1,0));axis.set_axis_off()
    axes[0].scatter(points[:,0],points[:,1],s=22,marker="+",color="white",label=f"Lower-mass-filtered points: {len(points)}")
    if crowded:
        q=points[sorted(crowded)]
        axes[0].scatter(q[:,0],q[:,1],s=90,facecolors="none",edgecolors="yellow",lw=1.2,label=f"Too-close candidates: {len(crowded)}")
    for a,b in pairs: axes[0].plot(points[[a,b],0],points[[a,b],1],color="yellow",lw=1,alpha=.8)
    axes[0].set_title("Grid and close-point conflicts (separation < half a grid spacing)")
    axes[0].legend(loc="upper left",facecolor="#222222",labelcolor="white")
    q=points[sorted(kept)]
    axes[1].scatter(q[:,0],q[:,1],s=32,facecolors="none",edgecolors="lime",label=f"Kept: {len(kept)}")
    if removed_by:
        q=points[sorted(removed_by)]
        axes[1].scatter(q[:,0],q[:,1],s=48,marker="x",color="red",label=f"Marked for removal: {len(removed_by)}")
    axes[1].set_title("Within close neighborhoods, keep points closest to the grid")
    axes[1].legend(loc="upper left",facecolor="#222222",labelcolor="white")
    fig.suptitle(f"Fourier grid: dx={spacing[0]:.2f}, dy={spacing[1]:.2f} px; translation=({offset[0]:.2f}, {offset[1]:.2f}) px; no rotation")
    fig.savefig(args.output/"grid_and_points.png",dpi=180);fig.savefig(args.output/"grid_and_points.pdf")
    plt.close(fig)
    print(json.dumps({"mass_retained":len(nodes),"mass_rejected":len(small),"spacing_xy":spacing.tolist(),
        "offset_xy":offset.tolist(),"too_close":len(crowded),"close_pairs":len(pairs),
        "kept":len(kept),"marked_for_removal":len(removed_by)},indent=2))


if __name__ == "__main__":
    main()
