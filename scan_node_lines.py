"""Score edge-to-edge axis-aligned lines by distance to retained crossing nodes."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=Path("output/crossing_nodes_lower_mass/nodes.json"))
    parser.add_argument("--model", type=Path, default=Path("output/color_grayscale_comparison/model.json"))
    parser.add_argument("--sigma", type=float, help="Gaussian distance tolerance in pixels; default is one measured string FWHM")
    parser.add_argument("--output", type=Path, default=Path("output/node_line_scans"))
    args = parser.parse_args()
    data = json.loads(args.nodes.read_text())
    model = json.loads(args.model.read_text())
    if Path(data["source"]).resolve() != Path(model["source"]).resolve():
        raise ValueError("Node and string-width data refer to different images")
    nodes = data["nodes"]
    points = np.array([[n["x"], n["y"]] for n in nodes])
    with Image.open(data["source"]) as image:
        width, height = ImageOps.exif_transpose(image).size
    sigma = model["width_fwhm"] if args.sigma is None else args.sigma
    eligible_x=np.array(["x" in n.get("scan_axes",["x","y"]) for n in nodes])
    eligible_y=np.array(["y" in n.get("scan_axes",["x","y"]) for n in nodes])
    x, vertical = line_scan(points[eligible_x,0], width, sigma)
    y, horizontal = line_scan(points[eligible_y,1], height, sigma)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, positions, scores in (("vertical", x, vertical), ("horizontal", y, horizontal)):
        np.savetxt(args.output/f"{name}_scan.csv", np.column_stack((positions,scores)), delimiter=",",
                   header="line_position_pixels,sum_of_gaussian_node_weights", comments="")
    (args.output/"scan.json").write_text(json.dumps({"source":data["source"],"nodes_file":str(args.nodes.resolve()),
        "node_count":len(nodes),"x_node_count":int(eligible_x.sum()),"y_node_count":int(eligible_y.sum()),"sigma_pixels":sigma,"measured_string_fwhm_pixels":model["width_fwhm"],
        "formula":"sum(exp(-0.5*(perpendicular_distance/sigma)^2))", "node_weights":"equal",
        "grid_or_fourier_constraints":False},indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex":False,"font.family":"DejaVu Sans"})
    fig, axes=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
    for axis, positions, scores, coordinates, color, title, xlabel in (
        (axes[0],x,vertical,points[eligible_x,0],"tab:blue","Vertical line spanning the full image height, scanned across x","Vertical line position x (pixels from left edge)"),
        (axes[1],y,horizontal,points[eligible_y,1],"tab:orange","Horizontal line spanning the full image width, scanned across y","Horizontal line position y (pixels from top edge)")):
        axis.plot(positions,scores,color=color,lw=1.6)
        axis.plot(coordinates,np.zeros(len(coordinates)),"|",color=color,alpha=.4,ms=7)
        axis.set(title=title,xlabel=xlabel,ylabel="Line score (effective nearby node count)",xlim=(0,positions[-1]),ylim=(0,None))
        axis.grid(alpha=.25)
    fig.suptitle(f"{len(nodes)} nodes (x: {eligible_x.sum()}, y: {eligible_y.sum()}); Gaussian distance tolerance sigma={sigma:.2f} px; no grid assumption")
    fig.savefig(args.output/"line_scans.png",dpi=180)
    fig.savefig(args.output/"line_scans.pdf")
    plt.close(fig)
    print(json.dumps({"nodes":len(nodes),"sigma_pixels":sigma,
                      "vertical_score_range":[float(vertical.min()),float(vertical.max())],
                      "horizontal_score_range":[float(horizontal.min()),float(horizontal.max())]}))


if __name__ == "__main__":
    main()
