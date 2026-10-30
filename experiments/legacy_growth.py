"""Select a confident corner grid, then jointly fit one complete side at a time."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, correlate
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.sparse import coo_matrix


from gridtracker import ridge_maps, color_evidence, estimate_string_width, crossing_evidence, sample, segment, find_seed, plot_width_diagnostics, plot_uncertainty

DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def add(key, direction, multiple=1):
    return tuple(a + multiple * b for a, b in zip(key, direction))


def extrapolate(key, nodes, steps):
    guesses = []
    for direction in DIRECTIONS:
        near, far = add(key, direction), add(key, direction, 2)
        if near in nodes:
            guesses.append(2 * nodes[near] - nodes[far] if far in nodes else
                           nodes[near] - np.asarray(direction) @ steps)
    return np.mean(guesses, axis=0)


def side_frontier(nodes, direction):
    """One complete row/column; a boundary-truncated row is never backfilled."""
    axis = 0 if direction[0] else 1
    edge = (max if sum(direction) > 0 else min)(key[axis] for key in nodes)
    return [add(key, direction) for key in sorted(nodes) if key[axis] == edge]


def fit_side(guesses, nodes, evidence, crossing, steps, smoothness, search_fraction, directional=None):
    """Simultaneously optimize every new node, including new-new edges and bends."""
    new_keys = sorted(guesses)
    keys = new_keys + sorted(nodes)
    index = {key: i for i, key in enumerate(keys)}
    count = len(new_keys)
    initial = np.array([guesses[key] for key in new_keys])
    fixed = np.array([nodes[key] for key in keys[count:]]).reshape(-1, 2)
    pitch = np.array([np.linalg.norm(steps[1]), np.linalg.norm(steps[0])])
    lower = np.maximum(initial - search_fraction * pitch, 0)
    upper = np.minimum(initial + search_fraction * pitch, np.array(evidence.shape[::-1]) - 1)
    # Fixed neighbors bound incoming progress; midpoint boxes preserve new-side order.
    for i, key in enumerate(new_keys):
        for direction in DIRECTIONS:
            neighbor = add(key, direction)
            dim = 0 if direction[1] else 1
            if neighbor in nodes:
                margin = 0.25 * pitch[dim]
                boundary = nodes[neighbor][dim] - margin if sum(direction) > 0 else nodes[neighbor][dim] + margin
            elif neighbor in guesses:
                boundary = (guesses[key][dim] + guesses[neighbor][dim]) / 2
                boundary += -0.01 if sum(direction) > 0 else 0.01
            else:
                continue
            if sum(direction) > 0:
                upper[i, dim] = min(upper[i, dim], boundary)
            else:
                lower[i, dim] = max(lower[i, dim], boundary)
    if np.any(lower >= upper):
        raise ValueError("No ordered search boxes remain for the new side")
    layout = np.vstack((initial, fixed))
    edges, triples = [], []
    for key in keys:
        for direction in ((0, 1), (1, 0)):
            neighbor, far = add(key, direction), add(key, direction, 2)
            if neighbor in index and (key in guesses or neighbor in guesses):
                edges.append((index[key], index[neighbor]))
            if neighbor in index and far in index and any(k in guesses for k in (key, neighbor, far)):
                triples.append((index[key], index[neighbor], index[far]))
    first, second, fractions, families = [], [], [], []
    for a, b in edges:
        n = max(5, int(np.linalg.norm(layout[a] - layout[b]) / 2))
        first.extend([a] * n)
        second.extend([b] * n)
        fractions.extend((np.arange(n) + 0.5) / n)
        families.extend([0 if keys[a][0] == keys[b][0] else 1] * n)
    first, second = np.array(first), np.array(second)
    fractions = np.array(fractions)[:, None]
    families = np.array(families)
    triples = np.array(triples, dtype=int).reshape(-1, 3)

    def residual(flat):
        points = np.vstack((flat.reshape(count, 2), fixed))
        along = (1 - fractions) * points[first] + fractions * points[second]
        bend = (points[triples[:, 0]] - 2 * points[triples[:, 1]] + points[triples[:, 2]]) / pitch
        values = sample(evidence, along)
        if directional is not None:
            for axis in (0, 1):
                mask = families == axis
                values[mask] = sample(directional[axis], along[mask])
        return np.r_[(1 - values) / np.sqrt(len(along)),
                     (1 - sample(crossing, points[:count])) / np.sqrt(count),
                     np.sqrt(smoothness) * bend.ravel() / np.sqrt(max(1, bend.size)),
                     (0.05 * (points[:count] - initial) / pitch / np.sqrt(count)).ravel()]

    start = initial.copy()
    for i in range(count):
        xx, yy = np.meshgrid(np.linspace(lower[i, 0], upper[i, 0], max(9, int(np.ceil(upper[i,0]-lower[i,0]))+1)),
                             np.linspace(lower[i, 1], upper[i, 1], max(9, int(np.ceil(upper[i,1]-lower[i,1]))+1)))
        candidates = np.vstack((np.clip(initial[i], lower[i], upper[i]),
                                np.column_stack((xx.ravel(), yy.ravel()))))
        # This only initializes the block; the final optimization moves all nodes together.
        scores = (1 - sample(crossing, candidates)) ** 2
        scores += smoothness * np.sum(((candidates - initial[i]) / pitch) ** 2, axis=1)
        start[i] = candidates[np.argmin(scores)]
    # A segment depends on its two endpoints, a bend on three nodes.
    dependencies = [np.column_stack((2 * first, 2 * first + 1, 2 * second, 2 * second + 1)),
                    np.arange(2 * count).reshape(count, 2)]
    bends = np.column_stack((2 * triples, 2 * triples + 1)).reshape(-1, 6)
    # Each vector bend contributes x and y residuals, in that order.
    dependencies.append(np.repeat(bends, 2, axis=0))
    dependencies.append(np.arange(2 * count)[:, None])
    row_indices, columns, offset = [], [], 0
    for block in dependencies:
        rows = np.repeat(np.arange(offset, offset + len(block)), block.shape[1])
        cols = block.ravel()
        variable = cols < 2 * count
        row_indices.extend(rows[variable])
        columns.extend(cols[variable])
        offset += len(block)
    sparsity = coo_matrix((np.ones(len(columns)), (row_indices, columns)),
                          shape=(offset, 2 * count)).tocsr()
    result = least_squares(residual, start.ravel(), bounds=(lower.ravel(), upper.ravel()),
                           jac_sparsity=sparsity, max_nfev=300, diff_step=1e-4,
                           ftol=1e-5, xtol=1e-5, gtol=1e-5)
    if not result.success:
        raise RuntimeError(f"Side fit failed: {result.message}")
    return dict(zip(new_keys, result.x.reshape(count, 2)))


def addition_uncertainty(nodes, new_keys, maps, crossing):
    """Score only local new edges/cells; established easy regions cannot dilute risk."""
    edges, scores = {}, {key: [] for key in new_keys}
    for key in nodes:
        for axis, direction in enumerate(((0, 1), (1, 0))):
            neighbor = add(key, direction)
            if neighbor in nodes:
                edge = key, neighbor
                edges[edge] = float(np.mean((1 - sample(maps[axis], segment(nodes[key], nodes[neighbor]))) ** 2))
    for (a, b), score in edges.items():
        for key in (a, b):
            if key in scores:
                scores[key].append(score)
    # A completed cell must be credible as a whole, not just at its corners.
    cells = {key: [] for key in new_keys}
    for r, c in nodes:
        quad = ((r,c), (r,c+1), (r+1,c+1), (r+1,c))
        if all(key in nodes for key in quad) and any(key in cells for key in quad):
            sides = [edges[quad[0],quad[1]], edges[quad[3],quad[2]],
                     edges[quad[0],quad[3]], edges[quad[1],quad[2]]]
            points = np.array([nodes[key] for key in quad])
            lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
            irregularity = 0.1 * (np.var(lengths[[0,2]] / np.median(lengths[[0,2]]))
                                  + np.var(lengths[[1,3]] / np.median(lengths[[1,3]])))
            uncertainty = float(np.mean(sides) + irregularity)
            for key in quad:
                if key in cells:
                    cells[key].append(uncertainty)
    return {key: max(float(np.mean(scores[key])), max(cells[key], default=0),
                     float((1 - sample(crossing, [nodes[key]])[0]) ** 2))
            for key in new_keys}


def grow_grid(rgb, cycles=None, width=None, smoothness=0.2, search_fraction=0.35, max_uncertainty=0.45):
    if cycles is not None and (not isinstance(cycles, int) or cycles < 0):
        raise ValueError("Cycles must be a nonnegative integer, or None to reach image boundaries")
    if not np.isfinite(smoothness) or smoothness < 0 or not 0 < search_fraction < 0.5:
        raise ValueError("Require smoothness >= 0 and 0 < search fraction < 0.5")
    if not 0 < max_uncertainty < 1:
        raise ValueError("Uncertainty threshold must be between zero and one")
    maps = ridge_maps(rgb)
    seed, selected, candidates = find_seed(maps, max_uncertainty)
    initial_seed = seed.copy()
    if width is None:
        width, width_samples = estimate_string_width(rgb, seed)
    else:
        if not np.isfinite(width) or width <= 0:
            raise ValueError("String width must be finite and positive")
        width_samples = []
    evidence = color_evidence(rgb, maps, seed)
    crossing = crossing_evidence(evidence, width)
    steps = np.array([np.mean(np.diff(seed, axis=0), axis=(0, 1)),
                      np.mean(np.diff(seed, axis=1), axis=(0, 1))])
    initial = {(r, c): seed[r, c].copy() for r in range(len(seed)) for c in range(len(seed))}
    # The same block optimizer also fits all seed nodes jointly, with none fixed.
    nodes = fit_side(initial, {}, evidence, crossing, steps,
                     smoothness, min(search_fraction, 0.12), directional=maps)
    seed = np.array([[nodes[r,c] for c in range(len(seed))] for r in range(len(seed))])
    evidence = color_evidence(rgb, maps, seed)
    crossing = crossing_evidence(evidence, width)
    steps = np.array([np.mean(np.diff(seed, axis=0), axis=(0, 1)),
                      np.mean(np.diff(seed, axis=1), axis=(0, 1))])
    snapshots, guesses_by_cycle, sides = [dict(nodes)], [], []
    limits = np.array(evidence.shape[::-1]) - 1
    safety_limit = int(np.ceil(4 * sum(limits / np.array([np.linalg.norm(steps[1]), np.linalg.norm(steps[0])]))))
    completed, active = set(), None
    rejected, accepted_scores, rejected_keys = [], {}, set()
    for _ in range(cycles if cycles is not None else safety_limit):
        options = []
        for direction, name in zip(DIRECTIONS, ("left", "right", "up", "down")):
            if name in completed:
                continue
            guesses = {key: extrapolate(key, nodes, steps) for key in side_frontier(nodes, direction)}
            guesses = {key: p for key, p in guesses.items() if key not in rejected_keys and np.all((p >= 0) & (p <= limits))}
            if guesses:
                dim = 0 if direction[1] else 1
                distance = np.mean([nodes[add(key, direction, -1)][dim] for key in guesses])
                if sum(direction) > 0:
                    distance = limits[dim] - distance
                options.append((distance, name, guesses))
            else:
                completed.add(name)
        if not options:
            break
        continuing = [option for option in options if option[1] == active]
        _, side, guesses = continuing[0] if continuing else min(options, key=lambda item: item[0])
        active = side
        accepted_guesses = guesses.copy()
        while accepted_guesses:
            proposed = fit_side(accepted_guesses, nodes, evidence, crossing, steps, smoothness, search_fraction)
            uncertainties = addition_uncertainty(nodes | proposed, proposed.keys(), maps, crossing)
            bad = [key for key, score in uncertainties.items() if score > max_uncertainty]
            if not bad:
                nodes.update(proposed)
                accepted_scores.update(uncertainties)
                break
            for key in bad:
                rejected.append({"row": key[0], "column": key[1], "side": side,
                                 "uncertainty": uncertainties[key]})
                rejected_keys.add(key)
                del accepted_guesses[key]
            # Refit survivors together, without rejected nodes pulling on them.
        if not accepted_guesses:
            completed.add(side)
            active = None
            continue
        guesses_by_cycle.append(accepted_guesses)
        sides.append(side)
        snapshots.append(dict(nodes))
    else:
        if cycles is None:
            raise RuntimeError("Grid growth exceeded its safety limit; inspect extrapolated spacing")
    return {"snapshots": snapshots, "guesses": guesses_by_cycle, "sides": sides, "seed": seed,
            "initial_seed": initial_seed, "width": width, "width_samples": width_samples,
            "crossing": crossing, "rejected": rejected, "accepted_scores": accepted_scores,
            "evidence": evidence, "smoothness": smoothness, "search_fraction": search_fraction,
            "max_uncertainty": max_uncertainty, "selected": selected, "candidates": candidates}


def draw_grid(rgb, nodes, added):
    image = ImageEnhance.Brightness(rgb).enhance(3)
    draw = ImageDraw.Draw(image)
    for key, point in nodes.items():
        for direction in ((0, 1), (1, 0)):
            neighbor = add(key, direction)
            if neighbor in nodes:
                color = "yellow" if key in added or neighbor in added else "cyan"
                draw.line([tuple(point), tuple(nodes[neighbor])], fill=color, width=2)
        x, y = point
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="yellow" if key in added else "white")
    return image


def plot_crossing_overlay(rgb, result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    score = result["crossing"]
    size = 2 * max(6, int(np.ceil(3 * result["width"]))) + 1
    fig, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    axis.imshow(ImageEnhance.Brightness(rgb).enhance(3))
    heatmap = axis.imshow(score, cmap="turbo", vmin=0, vmax=1, alpha=0.85 * score)
    axis.set_title(f"Crossing score overlay: {size} x {size} window, string FWHM {result['width']:.2f} px")
    axis.set_axis_off()
    # An opaque colorbar preserves the score legend independently of overlay alpha.
    legend = plt.cm.ScalarMappable(norm=heatmap.norm, cmap=heatmap.cmap)
    fig.colorbar(legend, ax=axis, shrink=0.8, label="Four-arm crossing score (heuristic)")
    fig.text(0.02, 0.02, "Image shown at 3x brightness. Opacity = 0.85 x score; zero score is fully transparent.", fontsize=10)
    fig.savefig(output / "crossing_overlay.png", dpi=180)
    fig.savefig(output / "crossing_overlay.pdf")
    plt.close(fig)
    np.save(output / "crossing_scores.npy", score)


def contact_sheet(panels, path):
    sheet = Image.new("RGB", (1280, 360 * ((len(panels) + 1) // 2)), "#121820")
    draw = ImageDraw.Draw(sheet)
    for i, (title, image) in enumerate(panels):
        x, y = (i % 2) * 640, (i // 2) * 360
        draw.text((x + 10, y + 10), title, fill="white", font=ImageFont.load_default(size=17))
        sheet.paste(ImageOps.contain(image, (628, 310)), (x + 6, y + 40))
    sheet.save(path)


def save_growth(rgb, result, output):
    output.mkdir(parents=True, exist_ok=True)
    panels, records = [], []
    guide = ["# Corner selection and directional growth", "", "Yellow: new segments/nodes. Cyan: accepted segments. White: accepted nodes.", ""]
    for cycle, nodes in enumerate(result["snapshots"]):
        previous = result["snapshots"][cycle - 1] if cycle else {}
        added = nodes.keys() - previous.keys()
        image = draw_grid(rgb, nodes, added)
        name = f"{cycle:02d}_{'seed' if cycle == 0 else 'growth'}.png"
        image.save(output / name)
        label = f"{result['selected']['corner']} seed" if cycle == 0 else f"{cycle}: {result['sides'][cycle - 1]}"
        title = f"{label}: {len(nodes)} nodes (+{len(added)})"
        panels.append((title, image))
        guide.extend([f"## {title}", "", f"![{title}]({name})", ""])
        records.append({"cycle": cycle, "side": None if cycle == 0 else result["sides"][cycle - 1], "nodes": [{"row": r, "column": c, "xy": p.tolist(),
                         "new": (r, c) in added} for (r, c), p in sorted(nodes.items())],
                        "guesses": [] if cycle == 0 else [
                            {"row": r, "column": c, "xy": p.tolist()}
                            for (r, c), p in sorted(result["guesses"][cycle - 1].items())]})
    contact_sheet(panels, output / "overview.png")
    final = result["snapshots"][-1]
    draw_grid(rgb, final, final.keys() - result["snapshots"][0].keys()).save(output / "partial_mesh.png")
    before = {(r,c): point for r, row in enumerate(result["initial_seed"]) for c, point in enumerate(row)}
    contact_sheet([("Initial seed", draw_grid(rgb, before, set())),
                   ("Optimized seed", draw_grid(rgb, result["snapshots"][0], set()))],
                  output / "seed_optimization.png")
    phases = [0] + [i for i in range(1, len(panels))
                    if i == len(panels) - 1 or result["sides"][i - 1] != result["sides"][i]]
    contact_sheet([panels[i] for i in phases], output / "phases.png")
    plot_uncertainty(result, output)
    plot_width_diagnostics(rgb, result, output)
    plot_crossing_overlay(rgb, result, output)
    corners = []
    for i, corner in enumerate(("top-left", "top-right", "bottom-left", "bottom-right")):
        records_here = [r for r in result["candidates"] if r["corner"] == corner]
        accepted = [r for r in records_here if r["accepted"]]
        if not records_here:
            corners.append((f"{corner}: no 3x3 candidate", rgb))
            continue
        record = accepted[-1] if accepted else records_here[0]
        xx, yy = np.meshgrid(record["xs"], record["ys"])
        mesh = {(r,c): np.array([xx[r,c], yy[r,c]]) for r in range(record["n"]) for c in range(record["n"])}
        image = draw_grid(rgb, mesh, set(mesh))
        label = f"{corner}: {record['n']}x{record['n']}, U/area={record['density']:.3f}"
        label += " accepted" if accepted else " rejected"
        corners.append((label, image))
    contact_sheet(corners, output / "corner_candidates.png")
    guide[4:4] = ["![Uncertainty by grid size](uncertainty.png)", "", "![Corner candidates](corner_candidates.png)", ""]
    (output / "README.md").write_text("\n".join(guide))
    Image.fromarray(np.uint8(result["crossing"] * 255)).save(output / "crossing_evidence.png")
    Image.fromarray(np.uint8(result["evidence"] * 255)).save(output / "color_evidence.png")
    (output / "growth.json").write_text(json.dumps({"smoothness": result["smoothness"], "width_fwhm": result["width"],
        "width_samples": result["width_samples"],
        "search_fraction": result["search_fraction"], "max_uncertainty": result["max_uncertainty"],
        "selected": result["selected"], "candidates": result["candidates"], "stages": records,
        "rejected": result["rejected"], "accepted_scores": [
            {"row": key[0], "column": key[1], "uncertainty": value}
            for key, value in result["accepted_scores"].items()]}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/confident_growth"))
    parser.add_argument("--cycles", type=int, help="Optional layer limit; default grows to all reachable edges")
    parser.add_argument("--width", type=float, help="Override automatically estimated apparent string FWHM, in pixels")
    parser.add_argument("--smoothness", type=float, default=0.2)
    parser.add_argument("--search-fraction", type=float, default=0.35)
    parser.add_argument("--max-uncertainty", type=float, default=0.45)
    args = parser.parse_args()
    with Image.open(args.image) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
    result = grow_grid(rgb, args.cycles, args.width, args.smoothness, args.search_fraction, args.max_uncertainty)
    save_growth(rgb, result, args.output)
    print("Estimated width (FWHM pixels):", result["width"], "profiles:", len(result["width_samples"]))
    print("Selected:", result["selected"])
    print("Directions:", result["sides"])
    print("Nodes by cycle:", [len(nodes) for nodes in result["snapshots"]])


if __name__ == "__main__":
    main()
