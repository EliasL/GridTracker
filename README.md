# GridTracker

Detect a tennis-racket string grid from dark, roughly axis-aligned images.
The reproducible current pipeline is `pipeline.py`; historical experiments are
under `experiments/` and previous artifacts remain under `output/`.

## Open by clicking

After the one-time Python/dependency setup below, open the launcher for your system:

- **Windows:** double-click `Start GridTracker.bat`.
- **macOS:** double-click `Start GridTracker.command`.
- **Linux:** open `Start GridTracker.sh` with your file manager's **Run as a
  program** action (enable executable permissions if your copy lost them).

Choose the folder containing the image frames, then click **Open GridTracker**.
Results and saved edits default to that same folder. Uncheck **Save results in the
same folder** to browse for another destination. Choose the same destination on
later runs to resume your saved work. Closing the chooser cancels without starting
analysis. Launchers use the project's `.venv` if available, otherwise system
Python 3. A copied virtual environment is not portable between operating systems;
install dependencies on each computer. Python must include Tk (on Linux this may
require your distribution's `python3-tk` package).

When input and output share a folder, the application excludes its reserved
`tracking_overview.png`, `calibration_image.png`, and `calibration_input.png` files
from frame discovery. Results in subfolders are never scanned as input frames.

One-time setup (Python 3.10 or later, with `uv` installed):

```sh
uv venv .venv
uv pip install --python .venv -r pyproject.toml
```

The folder chooser can also be launched with `python launch_gridtracker.py`.
Launcher errors remain in the terminal so they can be diagnosed.

## Command line

Run the application on a folder:

```sh
.venv/bin/python gridtracker.py Archive-2026-10-29
```

Results, review checkpoints, calibration, and CSVs go into
`output/Archive-2026-10-29/`. These command-line defaults remain unchanged. Any
image folder works; use `--output` to choose a different results directory
(including the input folder). `--fps 8000` is optional; the rate can also be entered
in the window. The default order is download/birth time, with filename ties.
`--order filename` uses natural numeric filename ordering. For a common filename
prefix followed by frame numbers, numbering gaps are preserved in timestamps,
tracking intervals, velocity, and acceleration. No images are visually inspected
automatically. Detection uses four single-threaded processes by default.

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r pyproject.toml
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_batch.py --workers 4
.venv/bin/python build_report.py
```

`run_batch.py` orders JPEGs by file birth/download time (mtime fallback), runs
independent frame processes, and sets all numeric-library thread limits to one
before importing NumPy/SciPy. Individual images use only single-threaded numeric
operations; parallelism is across frames. It records errors explicitly and fails
if any frame fails. No images are visually inspected automatically.

## Final process

1. Color-neutral directional contrast selects a strong corner seed, grown in
   square size until uncertainty exceeds 0.45. This seed only learns image
   characteristics; it is not the final fitted grid.
2. Fit isolated perpendicular profiles to estimate apparent Gaussian FWHM,
   pooling both directions. Learn string RGB chromaticity from these profiles.
3. Form a scalar string-color intensity image. Compute a width-matched four-arm
   crossing score, requiring all arms and penalizing off-axis support. The exact
   rectangular correlations are separable, avoiding large 2D convolutions.
4. Select the start of a sustained shallow log-survival slope: within 20% of the
   shallowest local slope, sustained for at least 5% of the tested range.
   If no sustained interval exists, the application keeps an explicitly flagged
   shallowest-slope estimate for temporal recovery/review. The standalone threshold
   helper still raises by default. Black color samples contain no chromaticity
   information and are discarded with counts recorded; too few usable samples fail.
5. One geometric centroid per eight-connected above-threshold region. Keep pixel
   mass >= mean minus population SD; there is no upper cutoff or weighted-mass
   criterion.
6. Bootstrap straight lines from x/y node proximity scans. Remove split maxima
   with the iterative three-peak median-spacing test, until a second pass changes
   nothing. X-rejected peak supporters are removed; y cleanup is peak-only.
7. Detect endpoints in all four border strips. The batch uses a histogram/Otsu
   threshold for edge proposals because several frames do not have a sustained
   shallow edge-score slope. Crossing thresholds still use the agreed slope
   criterion. Complete missing endpoints from nearby local edge/cross maxima.
   No peak inside the line interval is recorded as unsupported, never invented.
8. Add endpoints before final scans and maxima cleanup. Top/bottom endpoints
   contribute only to the x-scan; left/right endpoints only to y.
9. Find all local minima of each post-cleanup scan. Assign eligible node
   coordinates to the intervals between minima, with image edges closing outer
   groups. Compute population mean/SD once in each group. Remove nodes with
   |coordinate - mean| / SD >= 2 in either eligible direction. Undefined SD does
   not reject. No isolated-neighbor or global line-score rejection is applied.
10. Fit cubic Bezier curves with four fixed evenly spaced longitudinal control
    positions, four fitted transverse controls, robust node residuals, and zero
    bending penalty. Omit candidates with fewer than four distinct longitudinal
    node positions. Edge identities come from the bootstrap scaffold; assignments
    remain fixed during fitting.

## Results and inspection

`output/final_batch/manifest.json` records all results, input SHA256s, worker count
and failures. Each `frame_XX/` contains detailed plots, scalar/crossing arrays,
`result.json`, final nodes, rejected nodes, local statistics and curve controls.

`output/pdf/gridtracker_process_report.pdf` contains a batch summary and every
stage plot for every frame. Its companion validation JSON records page and image
bounds checks. Review the plotted results before treating them as measurements:
there are no hand-labelled accuracy metrics, and
normal rigid groups can flag real bending as a statistical outlier. Edge and
crossing scores are heuristic, not calibrated probabilities. Unsupported
endpoints and curves remain explicitly recorded.

## Temporal tracking and first-frame verification

```sh
.venv/bin/python run_temporal.py --workers 4
# Optional: prepare detections without opening a window, then run the above.
.venv/bin/python run_temporal.py --prepare-only --workers 4
# Reuse a previously explicitly approved reference, with a known sampling rate.
.venv/bin/python run_temporal.py --reference output/temporal/reference.json --fps 1000
```

Detector workers finish at nodes, before Bezier fitting, and keep all crossing
components (including spatial rejects) for temporal recovery. Frame zero is
submitted first. A Matplotlib/Tk editor opens as soon as it finishes, while other
single-threaded worker processes continue. The GUI uses the main process; the
detector's noninteractive plotting backend stays in workers.
First-frame setup, node editing, review, and length calibration share one figure.
Application runs omit the large diagnostic plot/array collection by default;
`--diagnostics` enables it. Existing `run_batch.py` reports remain available.

Drag to move; right-click or Delete removes a node. With the pointer over the
image, `a` adds a crossing and `e` adds an endpoint on the nearest border. `u`
undoes; `i` toggles `(row,column)` labels; `h` shows all candidates. Use `[`/`]`
to change the selected column and `,`/`.` for its row. Labels start at zero.
Zoom/pan uses the toolbar and disables editing until deselected. Faint connecting
lines help check string identities. Approval rejects duplicate identities,
reversed string order, and inconsistent boundary nodes. Closing saves an
unapproved draft and does not start propagation. Node moves, additions, deletions,
row/column edits, undo, and unfinished frame-rate text autosave atomically after
each action. Dragging also checkpoints every 500 ms and on mouse release. Drafts
preserve the selected node, labels, zoom, and ID allocator. **Approve and track** saves a
reference bound to the exact image pixels and detection results. Reruns reuse
matching detection caches and resume drafts.

Large hollow yellow circles suggest missing crossings. Current edge endpoints set
the grid dimensions; hints update while endpoints or crossings are edited. Each
string is interpolated through its existing nodes, and crossing predictions within
two string widths of the border are suppressed to tolerate clipped partial lines.
These are suggestions only: they never add a node or block confirmation. An added
endpoint can introduce a new string identity. Row/column labels are stable integer
identifiers, not necessarily consecutive spatial ranks after inserting a string.
The first editor has an optional **Frames / second** box, also available when
editing subsequent frames. Blank uses frame-based units.

Sequential tracking requires explicit first-frame approval and completion of all
independent detections. Each crossing keeps its ID and row/column identity. A
one-step velocity prediction for currently observed interiors initializes pyramidal Lucas-Kanade optical
flow on one fixed, first-frame string-color transformation. Forward/backward
agreement, patch correlation, one-to-one assignment, and grid order checks validate
observations. `--method position` selects a previous-position/detection baseline
without optical flow or velocity prediction. Thresholds in `TrackingConfig` are
heuristics, not calibrated probabilities.

Lost and new nodes follow different rules:

- An unmatched interior keeps its last observed position immediately. The old
  velocity is cleared, so subsequent searches stay at that location rather than
  drifting. It is marked `predicted` for three frames, then `lost`; its ID is retained.
  Uncertainty increases, but the search radius
  stays below half the grid spacing. Stronger patch evidence is required to recover
  after a gap. If motion is too uncertain, recovery can fail; it is reported.
- Spatially rejected detections can recover a track when appearance and motion
  agree. Ambiguous matches remain missing.
- An unmatched edge node remains `held` indefinitely at its last identified
  position. Only a reliable edge identification updates it. Held endpoints remain
  visible and available to editing, confirmation, grid hints, and curve fitting,
  and have their own gold legend entry. They are still marked unobserved; no fresh
  detection or physical velocity is invented. Explicit human deletions in confirmed
  frames remain respected.
- Unclaimed detections near known tracks are distinguished from new candidates.
  Candidates get temporary IDs, hit counts, and expire after two absent frames.
  Even persistent candidates **never become verified tracks automatically**.
  Additions in a human-confirmed frame expand the verified roster; crossings
  entering the image are not automatically admitted.
- Measured/manual positions and held edge endpoints support Bezier fits; missing
  interiors and unverified candidates do not. Curves still have four controls and no bending penalty.
  Boundary intersections retain string identity, but velocity/acceleration are
  null because they are not material points.

`output/temporal/tracking.json` and `tracked_XX.json` preserve observations,
predictions, filtered positions, uncertainty, match reasons, candidate lifecycle,
and curves. Fits use observed coordinates. The Kalman state predicts tracking
positions; reported/exported velocity and acceleration use finite differences of
consecutive measured positions, with second-order endpoint differences when
three positions are available. Missing observations split the calculation; two
positions permit velocity but not acceleration. Boundary intersections have no
physical velocity or acceleration. Differentiation can amplify localization noise.
Without `--fps` or an entered sampling rate,
units are pixels/frame and pixels/frame². Supply the sampling rate of these
exported frames, which may differ from the source video FPS; download times give
order only. Numbered frame gaps are accounted for when present. Motion is in image coordinates
and includes camera/racket motion; physical deformation needs calibration and any
desired subtraction of global motion.

After propagation, the interactive review window has a frame slider, a colored
node-status legend, visibility toggles, and clickable histories. Arrow keys change
frame. Existing first-frame approval is reused on restart; every subsequent frame
requires its own explicit confirmation. The workflow is:

1. Nodes are **always editable** beside the frame slider and colored legend.
   Drag to move, right-click/Delete to remove, and use `a`/`e` over the image to
   add a crossing/edge node. Instructions and explanations live in the optional
   **Help** box. The visibility checkboxes have explicit dark labels.
   Edits autosave as unconfirmed drafts, including when changing frames or closing.
   Blue marks are edited drafts; red rings identify conflicting grid identities.
   Curve previews refit as nodes move. Export includes saved drafts from all frames.
2. **Confirm frame** fixes the current observations and manual edits as a human
   reference and advances to the next unconfirmed frame. Held edges are included;
   missing interior guesses are excluded unless explicitly repositioned. Changes
   to an already confirmed frame remain drafts until confirmed again.
3. **Reanalyze all** tracks forward from human anchors and backward between anchors,
   improves unconfirmed frames, and refits curves. Forward/backward disagreements
   stay unobserved. Confirmed coordinates and identities are checked unchanged.
   A node absent in a confirmed frame remains recoverable in future frames.
   Confirmed frames can only change through another explicit human edit.
   **Reanalyze next 10** changes only the ten frames after the selected frame (or
   the remaining frames near the end), preserving confirmed frames and everything
   outside that range. It replays from the nearest earlier human reference for
   tracking context. Confirm edits before reanalysis to use them as references.
   Changing the sampling rate requires full reanalysis.
   **Unmark frame** removes the selected later frame's human constraint and
   immediately reruns automatic analysis. Its former anchor and editing draft are
   retained in `reviews/history/`; stale drafts do not override the new automatic
   result. The first frame stays confirmed as the required initialization reference.
4. **Export CSV** is always available, including during editing and before first
   approval. Exported nodes are **intersections of the fitted horizontal and vertical
   cubic Bezier curves**, rather than the draggable guide points. Draft edits are
   fitted before computing crossings. Existing displayed fits are otherwise used
   unchanged. Export never confirms frames or changes anchors. `nodes.csv` has one row
   per frame and row/column pair: coordinates, provenance, velocity,
   acceleration, units, and a JSON field preserving additional node information.
   IDs such as `r3:c7` identify the intersection of row 3 and column 7 across time.
   Velocity and acceleration are recomputed from those intersections, with the same
   frame timing and physical calibration. Edge endpoints are fitting constraints,
   rather than rows in the main node CSV. Intersections are marked derived
   (`measurement_source=bezier_intersection`); frame verification describes the
   guide data, not a direct human measurement of each exported point.
   Missing fits, no intersection, or multiple intersections produce explicit
   statuses and blank coordinates; editable points are never substituted.
   Missing/ambiguous cells break the corresponding motion calculation.
   Before first approval, completed independent detections are fitted, IDs are
   local to each frame (`id_scope=frame`), and motion estimates are omitted. Companion
   `nodes_curves.csv` contains Bezier controls and fit statistics;
   `nodes_candidates.csv` preserves the independent detector candidates.

**Calibrate length** uses the same window. Load an image at the video's
magnification, drag the endpoints of the yellow line (or drag its whole body),
enter its known length, and select `mm`, `cm`, `m`, `µm`, or `in` from the dropdown.
**Apply to video** saves the scalar length-per-pixel calibration and a durable copy
of the reference image in the output directory. The same scale applies to all
frames. Pixel coordinates and motion values remain in CSVs; additional
`x_physical`, `y_physical`, `velocity_physical_x/y`, and
`acceleration_physical_x/y` columns use the chosen length unit and the video's
time unit. Bezier control coordinates are also exported in physical units.
Calibration is independent of node confirmation and can be changed later without
moving any nodes. No physical units are invented if calibration is unset.

`review.json` persists the human anchors, sampling rate, and pending reanalysis.
`ui_state.json` preserves the current frame and edit/review mode. Reruns return to
that frame and reopen an unfinished edit automatically. Older sessions without
this cursor file recover the newest unfinished draft or most recent confirmed
frame. Unapproved editor drafts remain separate; autosaving never confirms a
frame. Confirmed frames and applied calibration remain preserved. Updated tracking
rules trigger reanalysis on restart while checking that confirmed coordinates
remain unchanged.

`tracking_overview.png` samples at most 20 frames; no large PDF is generated.
Future documentation should use one detailed example-frame PDF and compact batch
comparisons. No images are automatically visually inspected.

## Code layout

- `gridtracker.py`: folder-based entry point and contrast/seed/width/crossing helpers.
- `pipeline.py`: current scientific stage ordering and diagnostic figures.
- `run_batch.py`: process-parallel batch launcher, single-thread workers.
- `run_temporal.py`: parallel detection and approval-gated temporal orchestration.
- `reference_editor.py`: first-frame editing, identities, approval, and drafts.
- `temporal_tracking.py`: prediction, optical flow, assignment, loss/recovery,
  and conservative candidate lifecycle.
- `tracking_review.py`: compact overview and interactive track inspection.
- `review_session.py`: human anchors, review checkpoints, kinematics, and CSV export.
- `mesh_hints.py`: advisory missing crossings from the edited boundary grid.
- `calibration.py`: draggable reference line, unit dropdown, and persistent scale.
- `build_report.py`: landscape A3 PDF assembly and structural validation.
- Other focused modules provide tested thresholding, peak cleanup, edge
  completion, local grouping, and Bezier fitting helpers. Their small inspection
  CLIs remain available. `experiments/HISTORY.md` documents prior iterations;
  discarded growth, Fourier and isolated-node approaches are not in the final
  pipeline.
