"""Clean split x-peaks, remove their supporting nodes, then repeat the y analysis."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.signal import find_peaks

from local_peak_spacing import prune_split_peaks, sliding_spacing
from scan_node_lines import line_scan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=Path, default=Path("output/node_line_scans"))
    parser.add_argument("--output", type=Path, default=Path("output/cleaned_scan_peaks"))
    args = parser.parse_args()
    metadata = json.loads((args.scans/"scan.json").read_text())
    node_data = json.loads(Path(metadata["nodes_file"]).read_text())
    nodes = node_data["nodes"]
    original = np.loadtxt(args.scans/"vertical_scan.csv", delimiter=",", skiprows=1)
    raw_x, _ = find_peaks(original[:,1])
    clean_x, median_x, removals_x = prune_split_peaks(raw_x)
    removed_x = set(raw_x) - set(clean_x)
    points = np.array([[n["x"], n["y"]] for n in nodes])
    assignments = raw_x[np.argmin(abs(points[:,0,None]-raw_x[None,:]), axis=1)]
    eligible_x=np.array(["x" in n.get("scan_axes",["x","y"]) for n in nodes])
    eligible_y=np.array(["y" in n.get("scan_axes",["x","y"]) for n in nodes])
    keep_mask = np.array([not eligible or peak not in removed_x for eligible,peak in zip(eligible_x,assignments)])
    surviving = [node for node,keep in zip(nodes,keep_mask) if keep]
    rejected = [node for node,keep in zip(nodes,keep_mask) if not keep]
    with Image.open(metadata["source"]) as image:
        width, height = ImageOps.exif_transpose(image).size
    if len(original) != width:
        raise ValueError("x-scan extent does not match source image")
    y, signal_y = line_scan(points[keep_mask & eligible_y,1], height, metadata["sigma_pixels"])
    raw_y, _ = find_peaks(signal_y)
    clean_y, median_y, removals_y = prune_split_peaks(raw_y)
    args.output.mkdir(parents=True, exist_ok=True)
    scans = [("x",original[:,0],original[:,1],raw_x,clean_x,median_x,removals_x),
             ("y",y,signal_y,raw_y,clean_y,median_y,removals_y)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex":False,"font.family":"DejaVu Sans"})
    fig, axes = plt.subplots(2,2,figsize=(15,8),constrained_layout=True)
    summaries = {}
    for column, (name,positions,signal,raw,clean,median,removals) in enumerate(scans):
        rejected_peaks = np.array(sorted(set(raw)-set(clean)),dtype=int)
        _, window, centers, raw_spacing, _ = sliding_spacing(signal,peaks=raw,allow_missing=True)
        _, _, _, clean_spacing, counts = sliding_spacing(signal,peaks=clean,allow_missing=True)
        top, bottom = axes[:,column]
        top.plot(positions,signal,lw=1.3,color="tab:blue")
        top.plot(clean,signal[clean],"o",color="green",ms=5,label=f"Retained maxima: {len(clean)}")
        top.plot(rejected_peaks,signal[rejected_peaks],"x",color="red",ms=9,mew=2,label=f"Removed maxima: {len(rejected_peaks)}")
        top.set(ylabel="Node proximity score",title=f"x-scan: {eligible_x.sum()} contributing nodes" if name=="x" else f"y-scan after x cleanup: {(keep_mask & eligible_y).sum()} contributing nodes")
        top.legend()
        bottom.plot(positions,raw_spacing,color="gray",ls="--",label="Before peak cleanup")
        bottom.plot(positions,clean_spacing,color="tab:blue",label="After peak cleanup")
        bottom.axhline(median,color="purple",ls=":",label=f"Reference median {median:g} px")
        bottom.set(xlabel=f"Scan position {name} (pixels)",ylabel="Local mean peak spacing (px)",title=f"Quarter-image sliding window: {window:g} pixels")
        bottom.legend(fontsize=8)
        for axis in (top,bottom):
            axis.axvspan(0,window/2,color="gray",alpha=.1)
            axis.axvspan(len(signal)-window/2,len(signal)-1,color="gray",alpha=.1)
            axis.set_xlim(0,len(signal)-1)
            axis.grid(alpha=.2)
        np.savetxt(args.output/f"{name}_scan.csv",np.column_stack((positions,signal)),delimiter=",",header="position,score",comments="")
        np.savetxt(args.output/f"{name}_spacing.csv",np.column_stack((positions,centers,raw_spacing,clean_spacing,counts)),delimiter=",",header="position,window_center,raw_spacing,clean_spacing,clean_peak_count",comments="")
        summaries[name]={"raw_peaks":raw.tolist(),"retained_peaks":clean.tolist(),"reference_median":median,
                         "removals":removals,"window_pixels":window,"undefined_clean_spacing_positions":int(np.isnan(clean_spacing).sum())}
    fig.suptitle("Remove split peaks using median spacing; gaps in spacing curves mean fewer than two peaks in the full window")
    fig.savefig(args.output/"cleaned_scans.png",dpi=180)
    fig.savefig(args.output/"cleaned_scans.pdf")
    plt.close(fig)
    (args.output/"analysis.json").write_text(json.dumps({"source":metadata["source"],"sigma_pixels":metadata["sigma_pixels"],
        "removed_nodes":rejected,"nodes":surviving,"scans":summaries,
        "node_assignment":"x-eligible nodes: nearest original x maximum; remove if that peak is removed; y-only edge endpoints unaffected",
        "y_cleanup":"peak-only diagnostic; no additional 2D nodes removed"},indent=2)+"\n")
    print(json.dumps({"removed_x":sorted(map(int,removed_x)),"nodes_before":len(nodes),"nodes_after":len(surviving),
        "y_raw_peaks":raw_y.tolist(),"y_clean_peaks":clean_y.tolist(),"removed_y":[r['peak'] for r in removals_y]},indent=2))


if __name__ == "__main__":
    main()
