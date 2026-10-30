# GridTracker

Detect and track a roughly axis-aligned tennis-racket string grid through image
frames, review the nodes interactively, and export fitted string intersections.

## Setup and launch

1. **Extract the entire downloaded ZIP** into a writable folder, such as Documents.
   Keep all the project files together; do not run a launcher inside the ZIP viewer.
2. If needed, install **Python 3.10 or later with Tk support** from
   [python.org](https://www.python.org/downloads/). Python itself must be installed;
   the launcher handles the application libraries.
3. Open the launcher for your system:

- **Windows:** double-click `Start GridTracker (Windows).bat`.
- **macOS:** double-click `Start GridTracker (Mac).command`.
- **Linux:** run `Start GridTracker (Linux).sh` in a terminal (enable executable
  permissions if needed).

On first launch, GridTracker automatically creates `.venv`, sets up pip, and
installs the libraries listed in `requirements.txt`. **No manual pip commands or
uv installation are required.** Keep the terminal open while setup runs; the
first installation needs internet and can take a few minutes. Packages are
installed locally in the project, without changing the system Python libraries.
Later launches check the environment and reuse it without contacting the package
server unless dependencies changed or need repair.

Python's Tk graphical interface is checked before downloading libraries. Linux
may need its matching `python3-tk` and `python3-venv` system packages. On Windows
and macOS, use a Python installation with Tcl/Tk support. Tk cannot be installed
with pip. If these prerequisites are missing, the launcher explains what is
needed and keeps its terminal open. Once Python starts, detailed startup and
installation output is saved in **`gridtracker_startup.log`** beside the launchers;
include it when reporting a setup problem. An incomplete or copied `.venv` is reported explicitly; remove only
that folder and rerun setup, keeping your images and results.

Choose the folder containing your image frames. Results and saved edits default
to **`gridtracker_results/` inside that folder**, keeping them separate from the
images. Uncheck **Save results in a subfolder of the frames folder** to choose a
custom destination. Choose the same results destination to resume saved work.
Closing the chooser cancels without starting analysis.

The chooser remembers the previous input and output folders and restores them on
the next launch. **Browse** opens in the parent of the last selected folder, making
it easy to choose another video alongside it. Canceling a browse dialog keeps the
previous choice. These local preferences are stored in `.launcher_state.json`.

Launchers prefer a working project interpreter, otherwise an installed Python
3.10+ interpreter. Environments are created separately on each computer; do not
copy `.venv` between computers or include it in a distribution ZIP.

For manual setup on macOS/Linux, the equivalent commands are:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use `py -3 -m venv .venv`, then
`.venv\Scripts\python.exe -m pip install -r requirements.txt`.

The same application can be launched from the command line:

```sh
.venv/bin/python gridtracker.py Archive-2026-10-29
.venv/bin/python gridtracker.py Archive-2026-10-29 --order filename --fps 8000
.venv/bin/python gridtracker.py Archive-2026-10-29 --output /path/to/results
# Prepare detections without opening the editor:
.venv/bin/python gridtracker.py Archive-2026-10-29 --prepare-only
# Also save diagnostic plots and score arrays:
.venv/bin/python gridtracker.py Archive-2026-10-29 --diagnostics
```

`python launch_gridtracker.py` opens the folder chooser. `run_temporal.py` accepts
the same arguments as `gridtracker.py`. Omitting the input folder uses `testData`.
Both positional input and `--input` default to `<input>/gridtracker_results`.

Frames are discovered nonrecursively. Supported extensions are JPEG, PNG, TIFF,
and BMP. Download/birth time determines the default order, with mtime fallback
and filename ties; `--order filename` selects natural numeric filename order.
A common filename prefix followed by frame numbers preserves numbering gaps in
tracking intervals and timestamps. Download times give order, not sampling rate.
For explicitly selected legacy destinations sharing the input folder, the reserved
`tracking_overview.png`, `calibration_image.png`, and `calibration_input.png` files
are excluded from frame discovery.

Four single-threaded detector processes run by default (`--workers` changes the
count). Frame zero is submitted first so it can be edited while detection of the
remaining frames continues. `--method position` selects a previous-position
baseline instead of optical flow and velocity prediction.

## Editing and review

First-frame setup, node editing, frame review, and length calibration share one
window. First-frame approval is required before tracking. Later frames each have
an explicit confirmation state.

- Drag a node to move it; right-click or Delete removes it.
- With the pointer over the image, `a` adds a crossing and `e` adds an endpoint
  on the nearest border.
- **Command-Z**, **Ctrl-Z**, **z**, or **u** undoes the last node edit in the current
  editor. Undo covers moves, additions, deletions, and row/column edits. It is
  available during first-frame setup and subsequent review; history resets when
  changing frames or reopening the editor. Text-field focus keeps editing keys
  inside the text field.
- `[`/`]` changes the selected column; `,`/`.` changes its row. Labels start at
  zero and remain stable integer identities, including when inserting strings.
- `i`/`h` toggles labels/candidates in the first editor. Review has checkboxes for
  **Guide nodes**, **Grid crossings**, candidates, held/missing nodes, curves,
  labels, and missing-crossing hints. Grid crossings are fitted intersections,
  updated as you edit. Hide Guide nodes to view just the derived geometry;
  enable them again to edit.
- Use the slider or arrow keys to change frames. Select **Help** for instructions.
- Toolbar zoom/pan disables node manipulation until deselected.

Edits autosave atomically as unconfirmed drafts, including incomplete frame-rate
text, selection, labels, zoom, and the ID allocator. Dragging checkpoints every
500 ms and on release. Closing saves a draft without approving it. Approval checks
for duplicate identities, reversed string order, and inconsistent boundary data.
References are bound to image hashes and the detection results.

**Confirm frame** fixes the current observations and edits as a human reference,
then advances to the next unconfirmed frame. Changes to a confirmed frame remain
drafts until confirmed again. Held edge endpoints are included; missing interior
positions require an explicit edit to become observations.

**Reanalyze all** tracks from human anchors, using forward/backward agreement
between anchors. Confirmed positions and identities are checked unchanged.
**Reanalyze next 10** changes only the next ten frames (or remaining frames),
preserving confirmed frames and everything outside that range. Confirm edits
before reanalysis to use them as anchors. Changing the sampling rate requires
full reanalysis.

**Unmark frame** releases a later frame's human constraint and immediately reruns
tracking. Its former anchor and drafts remain in `reviews/history/`. The first
frame stays confirmed as the required initialization reference.

The review legend groups guide nodes into green **Tracked / confirmed**, blue
**Edited**, and orange **Held / missing**. White dots mark fitted **Grid crossings**.
Yellow rings mark **Suggestions**, including missing-crossing hints and unclaimed
candidates; red rings mark **Conflicts**. Detailed tracking statuses remain in the
saved data. Suggestions never add nodes or block confirmation. Missing-crossing
hints within two string widths of an image border are suppressed to tolerate
clipped lines.

## Motion, missing nodes, and curve fitting

Tracking uses one fixed first-frame string-color transformation. One-step velocity
prediction initializes pyramidal Lucas-Kanade optical flow; forward/backward
agreement, patch correlation, one-to-one assignment, and grid order validate
observations. Thresholds in `TrackingConfig` are heuristics, not probabilities.

An unmatched interior stays at its last observation and clears its velocity. It
is marked `predicted` for three frames, then `lost`, retaining its identity.
Recovery searches stay below half the grid spacing and require stronger evidence
after a gap. Spatially rejected detections can recover a track when motion and
appearance agree; ambiguous matches remain missing.

An unmatched endpoint remains `held` at its last identified position until a
reliable identification updates it. It supports editing and fitting but is marked
unobserved and has no physical velocity. Explicit human deletions remain respected.
Unclaimed detections receive temporary candidate IDs and expire after two absent
frames; they never become verified tracks automatically. Human-confirmed additions
expand the verified roster.

Each string is fitted with a cubic Bezier curve: four fixed, evenly spaced
longitudinal controls and four fitted transverse controls, robust residuals, and
no bending penalty. At least four distinct longitudinal node positions are
required. Observations, manual positions, and held endpoints support fits; missing
interiors and unverified candidates do not.

Enter **Frames / second**, or use `--fps`, for timed motion. Use the sampling rate
of the exported frames, which may differ from the original video FPS. Blank uses
pixels/frame and pixels/frame². Velocity and acceleration use finite differences
of contiguous measured positions; gaps split the calculation. Two positions
permit velocity, three permit acceleration and second-order endpoint differences.
Boundary intersections have no material-point velocity or acceleration. Motion
includes camera/racket motion, and differentiation can amplify localization noise.

## Export and calibration

**Export MP4** is a separate button from **Export CSV**, available during first-frame
setup and subsequent review. It saves `grid_crossings.mp4` in the results folder,
with white fitted-crossing dots over the same brightened source images used in
review. It includes saved drafts without confirming frames, and creates no CSVs.
Missing or ambiguous intersections have no marker; guide points are never used
as substitutes. Before first approval, only completed independent detections are
available for export, with the current edits applied to frame zero.

The MP4 is a **30 fps preview**, with one output frame per available source image;
numbering gaps are not filled. This playback speed is separate from the sampling
rate used for velocity and acceleration. Odd image dimensions are padded by one
pixel on the right/bottom for the MP4 encoder, preserving all image coordinates.
The encoder uses the existing OpenCV dependency. Failed exports leave any existing
video intact.

**Export CSV** includes saved drafts and never confirms a frame. `nodes.csv`
contains intersections of the fitted horizontal and vertical Bezier curves,
rather than the draggable guide points. IDs such as `r3:c7` preserve row/column
identity across frames. Coordinates, provenance, velocity, acceleration, units,
and additional node metadata are exported. Missing or ambiguous intersections
produce explicit statuses and blank coordinates, breaking motion calculations.

`nodes_curves.csv` contains Bezier controls and fit statistics;
`nodes_candidates.csv` preserves independent detector candidates. Before first
approval, completed detections can be exported with IDs local to each frame and
without motion estimates. Edge endpoints constrain fits rather than appearing
as rows in the main node CSV. Exported crossings are derived measurements;
verification describes their guide data.

**Calibrate length** loads an image at the video's magnification. Drag the yellow
line's endpoints (or its whole body), enter its known length, choose `mm`, `cm`,
`m`, `µm`, or `in`, and select **Apply to video**. The saved scalar scale applies
to every frame. CSVs retain pixel values and add physical coordinate, velocity,
acceleration, and Bezier-control columns. Calibration is independent of node
confirmation; no physical units are invented when it is unset.

## Detection and diagnostic files

The scientific stages in `pipeline.py` are:

1. Select a corner seed using color-neutral directional contrast, growing it
   until uncertainty exceeds 0.45. It learns image characteristics, not final geometry.
2. Estimate Gaussian FWHM from isolated perpendicular profiles in both directions,
   then learn string RGB chromaticity. Black samples are counted and discarded;
   insufficient usable samples raise an error.
3. Construct a scalar string-color image and a measured-width four-arm crossing
   score with off-axis support penalized. Rectangular correlations are separable.
4. Select the start of a sustained shallow log-survival slope, within 20% of the
   shallowest slope over at least 5% of the range. When unsustained, explicitly
   flag the shallowest-slope estimate for temporal recovery and review.
5. Use one geometric centroid per eight-connected above-threshold component.
   Retain pixel mass at least mean minus population SD; there is no upper cutoff.
6. Bootstrap scan maxima and iteratively remove split peaks using median spacing.
   X-rejected supporters are removed; y cleanup changes peaks only.
7. Propose endpoints in all four border strips with a histogram/Otsu threshold.
   Complete missing endpoints from nearby edge/crossing maxima; unsupported
   intervals are recorded without inventing a point.
8. Add endpoints before final scans: top/bottom vote in x, left/right in y.
9. Group coordinates between scan minima and reject nodes at least two population
   SD from their own group mean in either eligible direction. Statistics are
   computed once; undefined SD does not reject. No isolated-node or global
   line-support filter is applied.
10. Fit curves from the retained nodes. Temporal detection stops before fitting;
    persistent identities determine the later fits.

Normal application runs skip diagnostic figure construction and score-array
exports. `--diagnostics` enables per-frame stage PNGs, width diagnostics, scalar
and crossing-score arrays in `frame_XX/`. Score thresholds and geometric models
are heuristic; there are no hand-labeled accuracy metrics.

## Saved work and development

A results folder contains detection caches (`detections.json`, `frame_XX/`),
tracking results (`tracking.json`, `tracked_XX.json`, `tracking_overview.png`),
human references and drafts (`reference.json`, `reviews/`), the review state
(`review.json`), calibration and its reference image, the interface cursor
(`ui_state.json`), and CSV exports. **Keep the complete results folder when
backing up or moving saved work.** Images are referenced by absolute path and hash.

Existing results formerly mixed into `Archive-2026-10-29/` were moved into its
`gridtracker_results/` subfolder. Older independent sessions under `output/` remain
available with `--output`; they were not merged into the newer session.

The initial Git commit preserves retired growth/Fourier/filter experiments,
standalone stage commands, and the older batch/PDF reporting workflow.

- `gridtracker.py`, `launch_gridtracker.py`: command-line and native chooser entry points.
- `bootstrap_gridtracker.py`: automatic local environment setup and startup diagnostics.
- `run_temporal.py`: detection caching, workers, and tracking orchestration.
- `detection.py`, `string_color.py`, and focused scientific modules: image analysis.
- `pipeline.py`: scientific stage ordering and optional diagnostics.
- `temporal_tracking.py`: prediction, assignment, and loss/recovery.
- `session_data.py`: identities, validation, hashes, and atomic JSON persistence.
- `node_editor.py`, `reference_editor.py`, `tracking_review.py`: shared editing and review.
- `review_session.py`: human anchors, reanalysis, kinematics, and CSV exports.
- `video_export.py`: MP4 previews of fitted crossings.
- `mesh_hints.py`, `calibration.py`: advisory gaps and physical scale.

Run the tests without opening a GUI:

```sh
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python -m unittest discover -s tests -v
```

Tests cover scientific helpers, a real-frame detector regression, optional
artifact generation, temporal recovery, confirmed anchors, exports, calibration,
and Matplotlib input events. Images are checked numerically/structurally, without
visual inspection.
