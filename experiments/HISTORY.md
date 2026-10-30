# GridTracker: corner seeds and joint side growth

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r pyproject.toml
.venv/bin/python gridtracker.py 'testData/WhatsApp Image 2026-10-28 at 12.13.25.jpeg'
.venv/bin/python -m unittest discover -s tests -v
```

This experiment works on one still image, with roughly axis-aligned strings.
No string hue is assumed. Luminance contrast detects light or dark strings;
colors are learned from the selected seed.

## Corner seed selection

1. Estimate rough pitch from a strongly repeating pattern anywhere in the image.
2. For each corner, project evidence through the adjacent outer third of the
   image to locate horizontal and vertical strings. Suppress duplicate peaks
   using the rough pitch. Corner grids start at the outermost detected strings,
   not at the literal image boundary.
3. Evaluate 3x3, 4x4, 5x5, ... crossings, expanding inward from each corner until
   available detected lines are exhausted. Larger candidates are evaluated for
   the diagnostic graph even after acceptance has stopped.
4. Accept only the prefix before the first candidate exceeding
   `--max-uncertainty`. Keep the largest accepted grid from each corner.
5. Select the retained corner grid with the smallest uncertainty per mesh area;
   equal scores favor larger area. An exact tie uses fixed corner order.

For each candidate edge, evidence deficit is the mean of `(1 - directional
contrast evidence)^2` along the straight segment. Each cell's uncertainty is the
average deficit of its four edges plus an irregular-spacing penalty:
`0.1 * (variance(dx / median(dx)) + variance(dy / median(dy)))`.

**Total uncertainty** is the sum of cell uncertainty times cell area, and
**uncertainty per area** is this total divided by the covered mesh area. This
normalization lets differently sized meshes be compared without automatically
favoring small meshes. It is a heuristic evidence score, not a probability,
confidence interval, or fitted coordinate covariance. The default threshold 0.45
is exploratory, chosen for inspecting the supplied image; it is not calibrated.
Color evidence is not used to score its own training seed.

## Seed refinement and conservative growth

Estimate string width from Gaussian-plus-linear-background fits to perpendicular
luminance profiles at isolated points along seed segments, away from crossings.
Both bright and dark contrast are fitted. Reject poor fits and pool both string
directions using the median Gaussian FWHM (full width at half maximum). This is
apparent image width including blur, not physical string diameter. The rough
contrast maps use their existing 2 px bootstrap scale, so width estimation does
not change the initial corner selection.

String/background colors are learned from supported pixels on seed strings and
background pixels inside the seed, using smoothed histograms in square-root RGB.
The width sets a four-arm crossing detector: normalized correlations measure each
arm separately, take the weakest arm, and subtract support in the four off-axis
quadrants. A broad bright patch or one isolated line should not look like a cross.
The filter assumes locally near-axis-aligned crossings; it is not rotation invariant.

First optimize **all seed nodes together**, with small search boxes (12% of pitch
or the configured search radius, whichever is smaller), directional segment
evidence, crossing detection, and smoothness. Then relearn the color model on the
refined seed. The old pointwise product of horizontal/vertical contrast was
removed: at a true intersection, the perpendicular string can suppress the local
directional contrast response right at the center.

Among unfinished sides with at least one in-image extrapolated crossing, choose
the nearest image edge by mean pixel distance. Grow toward that edge until no
further expected crossing lies inside the image, then choose the next nearest
unfinished edge. A completed direction stays completed. The initial top/right
image margins may already be too narrow to contain another string.

Each growth step proposes **one full row or column**, extrapolating from the last
two established crossings along each incoming string. Out-of-image guesses are
skipped entirely. A side can therefore be truncated at the boundary; truncated
portions are not later backfilled. Previously accepted nodes remain fixed. Growth is now a **conservative partial
pass**: optimize a tentative side, evaluate its local evidence uncertainty, and
remove all new nodes exceeding the threshold. Refit survivors jointly without
rejected nodes influencing them, repeating until stable. Rejected crossings are
not retried in this pass; another side can still be attempted. No optimizer
failure is silently treated as a low-confidence detection.

**All new nodes of that side are optimized simultaneously in one least-squares
problem.** The objective includes:

- Learned-color evidence along every new segment, including those **between new
  nodes** and those joining the established grid.
- Color support in both directions at each new crossing.
- Second-difference smoothness for every three-node sequence touching a new node,
  both **along the new side** and **continuing into the established grid**.
- A weak penalty for displacement from the extrapolated guess.

Coarse per-node initialization only selects starting positions. The subsequent
fit updates all new coordinates together; it is not a sequence of final per-node
fits. Sparse derivatives encode the shared endpoint and three-node dependencies.
Bounds enforce progress and ordering without allowing node identities to swap.

For a new crossing, acceptance uses the worst of: mean incident directional-edge
deficit, adjacent completed-cell deficit with opposite-edge spacing consistency,
and squared deficit of the four-arm crossing score. The same threshold as seed
selection is used. This prevents the already-certain seed area from hiding an
uncertain addition. It is intentionally more local and conservative than an
area-averaged whole-mesh score; it is still a heuristic, not a probability.

## Options

- `--max-uncertainty 0.45`: maximum accepted seed uncertainty per unit area.
- `--cycles N`: optional limit on added rows/columns. Default: attempt reachable sides until none can grow confidently. Zero exports seed selection without growth.
- `--smoothness 0.2`: bend coefficient; zero disables it. Its normalization now
  applies to the jointly fitted side, so values are not directly comparable to
  the previous independent-node objective.
- `--search-fraction 0.35`: per-coordinate radius relative to pitch, between 0
  and 0.5, further restricted by ordering constraints.
- `--width VALUE`: optional apparent-FWHM override; default estimates it automatically.
- `--output output/confident_growth`: output directory.

## Inspection files

- `partial_mesh.png`: final confidently accepted mesh. Refined seed is cyan;
  accepted additions are yellow. This is the main output for the current pass.
- `seed_optimization.png`: original versus jointly refined seed.
- `width_diagnostics.png` / `.pdf`: horizontal/vertical width distributions, pooled
  median, a grid zoom and width bar on an actual string, plus the fitted profile.
  Width inspection views use 6x brightness; fits use original pixel values.
- `width_measurements.csv`: accepted widths, directions, sample positions and errors.
- `crossing_evidence.png`: the four-arm matched-filter score.
- `crossing_overlay.png` / `.pdf`: transparent crossing-score heatmap over the
  image (opacity = 0.85 x score). `crossing_scores.npy` stores unquantized scores.
  The window spans +/- three estimated string widths, giving 49x49 pixels for
  the current 7.83 px width. Arm thickness follows that measured width. Four arm
  correlations must all respond, and off-axis background support is subtracted;
  this is a matched-filter combination, not one signed convolution alone.

- `uncertainty.png` / `uncertainty.pdf`: four corner curves for total uncertainty
  and uncertainty per area, with threshold, accepted sizes, and selected seed.
- `uncertainty.csv`: exact plotted numerical values.
- `corner_candidates.png`: largest accepted seed in each corner, or its rejected
  3x3 candidate when none qualifies.
- `00_seed.png`, `01_growth.png`, ...: selected seed and each new row/column.
  Newly added segments/nodes are yellow; accepted segments cyan and nodes white.
  Brightness is increased threefold for display only.
- `overview.png`: all steps; `phases.png`: seed and end of each directional phase.
- `color_evidence.png`: the learned color score, not a calibrated probability.
- `growth.json`: seed scores, threshold, every crossing, direction, and in-image
  extrapolated guesses, accepted scores, rejected crossings, and width measurements. Coordinates use the EXIF-oriented image, origin upper left.
- `README.md`: full-resolution image sequence.

## Initial result and limits

On the first supplied frame, threshold 0.45 selects the **upper-right 4x4 seed**:
uncertainty per area is approximately **0.394**, while 5x5 reaches **0.460** and is
rejected. The conservative pass now retains 30 crossings rather than forcing coverage of the
whole image. The width estimate is 7.83 px pooled median, from 54 accepted
profiles (horizontal: 33, median 7.44 px; vertical: 21, median 8.33 px). These
profiles are correlated observations from the seed, not independent samples.

The partial pass deliberately omits low-evidence regions. Local averaging and
thresholding can still accept a false match or reject a real faint string; neither
coverage nor measured accuracy is guaranteed. Width constancy is an approximation
in blurred/perspective images, and a Gaussian is an approximation to the true
cross-section. Missing initial lines, wrong pitch/identities, strong rotation,
and colors changing outside the seed remain limitations. Incompatible constraints,
insufficient measurements/evidence, a failed optimizer, or no accepted corner
seed raise errors instead of silently returning a fallback.

Tests cover seed threshold selection, uncertainty normalization, color-neutral
strings, simultaneous smoothness between new nodes, nearest-edge scheduling,
fixed accepted nodes, in-image guesses, synthetic positions, and invalid inputs. Additional tests check the cross detector against lone lines
and flat bright patches, width recovery on a known Gaussian grid, and rejection
of uncertain growth.

## Width-profile color grayscale comparison

```sh
.venv/bin/python compare_crossing_inputs.py 'testData/WhatsApp Image 2026-10-28 at 12.13.25.jpeg'
```

This preserves the current overlay, raw scores, input preview and metadata in
`output/color_grayscale_comparison/previous/`. An existing snapshot is never
overwritten. It verifies the old raw score against a reconstruction before
comparing anything.

The previous input is already a scalar string/background histogram likelihood.
The experimental new grayscale input instead samples string-core RGB values from
the same accepted width profiles (three samples inside each fitted FWHM). It
learns median RGB chromaticity and its spread, then calculates:

`grayscale = intensity along learned RGB color * chromaticity similarity`

Intensity is normalized by the median training-string intensity and clipped to
0..1. Similarity is Gaussian in normalized RGB chromaticity, with its scale
estimated from training samples. No string hue is hardcoded. The same small
smoothing and the exact same 49x49 crossing filter are used. The comparison keeps
input/score scales, original-image brightness and overlay opacity identical.

The four-panel `comparison.png`/`.pdf` shows the two inputs and their crossing
heatmaps side by side. `grayscale.png`/`.npy`, `crossing_scores.npy` and `model.json`
provide the new input, full-precision scores and fitted parameters. The original
main pipeline is unchanged; this is an isolated comparison. On the supplied
image the new input suppresses ball/background responses substantially, but also
reduces crossing strength. No labelled accuracy comparison has been made.

## Automatic crossing threshold

```sh
.venv/bin/python crossing_nodes.py
```

The new positive-score map is thresholded with strict `score > t`. Coverage uses
the entire image as denominator; zero scores are excluded. The first panel in
`output/crossing_nodes_auto/threshold_coverage.png` plots retained fraction on a
logarithmic y-axis. The second plots the magnitude of its smoothed log-slope.

Choose the first sustained interval within 20% of the shallowest positive slope.
Its span must be at least 5% of the tested threshold range. Slope estimation uses
512 uniformly spaced thresholds and a local straight-line fit (Savitzky-Golay, degree 1) in a window
spanning approximately 8% of that range. The sparse tail below 0.01% of image
pixels (at least 20 pixels) and smoothing-window endpoints are excluded from the
slope search. This is a tunable heuristic for the start of the gentle middle
region, rather than a three-line breakpoint fit, which selected an earlier knee.

For this image the chosen threshold is 0.0804, the shallow-slope interval extends
to 0.2082, and the shallowest slope occurs at 0.1729. It retains 1.7597% of the
image and yields 111 eight-connected components. Each node is the component's
uniform geometric pixel centroid; there is no size filtering. Numerical curves
and selection parameters are exported to CSV/JSON. `--threshold` overrides the
automatic choice; `--slope-tolerance` adjusts the 20% allowance. Earlier manual
threshold previews remain in their original directory.

### Component-mass filtering

`crossing_nodes.py` now defaults to `--mass-sigma 1` and writes to
`output/crossing_nodes_mass_filtered/`, preserving the previous unfiltered preview.
Mass is the number of pixels in each thresholded connected region. Compute mean
and population standard deviation over all regions once (no iterative clipping),
then retain only `mean - SD <= mass <= mean + SD`. Both the rendered region and
its centroid marker are removed when excluded. The threshold-coverage graph
continues to describe score thresholding alone, before the mass filter.

For the current threshold, 111 regions have mean mass 125.00 pixels and SD 72.45
pixels. The inclusive accepted interval is 52.55..197.45 pixels (integer sizes
53..197), retaining 62 regions and excluding 49. `nodes.json` stores the filter
statistics, retained nodes and rejected nodes; `mask.png` includes only retained
regions.

### Score-weighted component mass

The default is now `--mass-measure score`: a component's mass is the **sum of
crossing scores over its above-threshold pixels**. Apply the same one-pass
population mean +/- one standard deviation rule to these masses. Geometric
centroid positions and the crossing threshold are unchanged. Each JSON node
retains both `pixels` and `score_mass` for comparison. Use `--mass-measure pixels`
to reproduce area-only filtering. The new default output directory is
`output/crossing_nodes_weighted_mass/`; earlier area-filtered results remain
unchanged.

## Fourier grid from component centers

```sh
.venv/bin/python fit_point_grid.py
```

The mass filter now defaults to pixel count with **only the lower mean-SD cutoff**.
There is no upper mass limit, retaining 84 of the original 111 regions. The
`crossing_nodes.py` command also uses this rule; `--upper-mass-limit` restores the
old two-sided rule. Its new output is `output/crossing_nodes_lower_mass/`.

`fit_point_grid.py` reconstructs this selection from the unfiltered component
centers. For each coordinate axis it evaluates the direct Fourier transform of
unit point impulses, `abs(mean(exp(-2*pi*i*f*coordinate)))`. Each retained point
gets one vote; pixel masses do not reweight this spectrum. The search excludes DC
and periods longer than one third of the image extent, and by default considers
periods down to 8 pixels. The strongest interior peak is refined continuously.
Missing points are allowed. This is a point-coordinate Fourier transform, rather
than an FFT of the photograph. Harmonics and curved/perspective grids can produce
competing peaks.

Hold the selected spacings fixed and fit translation only, separately on each
axis, by minimizing mean squared distance to the nearest grid line. Globally
scan one period before local refinement, so a phase-boundary wrap is handled.
The fitted grid has no rotation. On the supplied frame, spacings are 93.85 px in
x and 71.06 px in y, with offsets (72.73, 29.70) px modulo their periods. The y
spectrum also has a substantial peak around 81.9 px; the rigid grid is ambiguous
in that direction and cannot represent the bent strings exactly.

Two points are flagged as too close when
`sqrt((dx/grid_pitch_x)^2 + (dy/grid_pitch_y)^2) < 0.5`.
`--min-separation` changes this fraction. Sort points by their Euclidean pixel
distance to the nearest grid intersection, and retain a point only when it is
not too close to an already retained point. A rejected competitor therefore loses
to a point closer to the grid. This avoids collapsing an entire chain of nearby
points when its endpoints are sufficiently separated. No additional global
cutoff on grid distance is applied: the ranking resolves close competitors.

`output/fourier_point_grid/grid_and_points.png` shows the translated grid and
conflicts (yellow rings), then retained points (green) and points marked for
removal (red crosses). The current pass identifies one close pair, keeps its
better-aligned point, and retains 83 of the 84 mass-filtered points.
`fourier_spectra.png`/`.pdf` show both spectra. `grid.json` stores spacing,
translation, alternative peaks, mass exclusions, close-pair IDs, per-point grid
distances and retention decisions. Prior image outputs are preserved.

## Scan full-length lines across the node positions

```sh
.venv/bin/python scan_node_lines.py
```

Use the 84 lower-pixel-mass-filtered points **before** Fourier-based duplicate
removal. A vertical line at x receives `sum(exp(-0.5*((x-node_x)/sigma)^2))`;
a horizontal line at y uses the same formula on y coordinates. Each node has equal
weight, with maximum contribution one. The lines span the full image, so only
perpendicular distance matters. No pitch, phase, rotation, or grid fitting is
involved. The default distance tolerance sigma is one estimated string width
(7.83 pixels), a smoothing choice; it can be changed with `--sigma`.

`output/node_line_scans/line_scans.png`/`.pdf` show vertical-line scores as a
function of x and horizontal-line scores as a function of y. The small tick marks
at zero show node coordinates. Each scan also has a CSV, and `scan.json` records
its parameters and source data.

## Sliding local maxima spacing

```sh
.venv/bin/python local_peak_spacing.py
```

This takes every local maximum of the longer x-scan without prominence or minimum
distance filtering. At each x, average the spacings between consecutive maxima
inside a centered window spanning 1/4 of the image width (320 pixels here). Both
endpoints of a spacing must lie inside the window. Near the image edges, clamp
the window center to the nearest position permitting a full window; do not shrink
the window. Fewer than two maxima in a window raises an error.

`output/local_peak_spacing/local_spacing.png`/`.pdf` show all 15 detected maxima
and the local mean spacing. Gray bands mark the edge regions with constant values.
`spacing.csv` and `spacing.json` contain the numerical values and conventions.

## Remove maxima that split a normal gap

```sh
.venv/bin/python clean_scan_peaks.py
```

Initialize the reference spacing to the median consecutive-maximum gap, and recompute it after every deletion.
For each triple with gaps `a,b`, compare `abs((a+b)/2 - median)` before deletion
with `abs(a+b - median)` afterward. Delete the middle maximum only if both
original gaps are shorter than the reference and deletion improves agreement.
Choose the largest improvement first, recalculate neighboring triples, and repeat;
recompute the reference median and retry all triples until none qualifies. A second complete pass is then idempotent. This implements the
proposed triple rule with an extra short-gap safeguard. It assumes most initial
gaps represent genuine spacing; substantial pitch changes or many false/missing
peaks can invalidate that assumption.

In x, this removes maxima at 98 and 202 px, leaving 13. Assign each of the original
84 two-dimensional nodes to its nearest original x maximum, then remove nodes
assigned to the rejected maxima: four nodes, leaving 80. This association is an
explicit assumption, not inferred string identity.

Recompute the y-scan from those 80 nodes using the unchanged Gaussian tolerance.
Apply the same triple rule diagnostically to its maxima: remove 154 and 245 px,
leaving 8. No additional 2D nodes are deleted by this y-only peak diagnostic.
`output/cleaned_scan_peaks/cleaned_scans.png`/`.pdf` compare both directions, with
retained/removed peaks and raw/clean local spacing. Quarter-image windows remain
320 px for x and 154 px for y, with nearest-full-window edge behavior. Some y
windows contain fewer than two cleaned peaks: their spacing is undefined and
shown as a gap, exported as NaN in CSV. `analysis.json` records all deletions,
reference spacings, node assignments, and remaining nodes. Earlier results remain
unchanged.


## Low-freedom cubic Bezier fits

```sh
.venv/bin/python fit_bezier_curves.py
```

Fit one curve for every cleaned maximum: 13 vertical and 8 horizontal curves.
Each is a single cubic Bezier with four control points. Their longitudinal
positions are fixed at the image boundaries and 1/3, 2/3 of the extent; only the
four transverse coordinates are optimized simultaneously. The two inner controls
are handles and generally do not lie on the curve. With these equally spaced
longitudinal controls, the curve is a cubic polynomial in x (horizontal strings)
or y (vertical strings), without backward loops along that axis.

Each detected node is assigned to the nearest cleaned maximum in the transverse
coordinate. This assignment remains fixed. Fit with robust soft-L1 node-distance
loss and a separate quadratic second-difference penalty on control coordinates
(`--smoothness`, now zero by default). A single Bezier is intrinsically smooth; the penalty adds a
preference for less bending. This is not a multi-frame fit or a joint constraint
between separate curves.

The bottom horizontal curve has only one assigned node and is now ignored.
Without a bend penalty, four distinct longitudinal node positions are required
for four free parameters; underconstrained curve candidates are omitted and
recorded under `ignored` in the output JSON. With a positive penalty, two distinct
positions suffice. An invalid direct fit raises an error.

`output/bezier_no_bending/curves.png`/`.pdf` overlay the 20 supported curves and 80 detected nodes;
`curves.json` records control points, assignments, fit modes, omissions, and residuals.
The prior penalized result remains in `output/bezier_curves/` for comparison.
Synthetic tests verify cubic recovery, bend suppression, sparse-node behavior,
and removal of multiple inserted maxima until a second cleanup pass changes
nothing. Another option, if all four adjustable points must lie *on* the curve,
is a cubic interpolating spline through four evenly spaced anchors; those are
not the same as Bezier handles.

## Border endpoints and the straight cleaned-maxima grid

```sh
.venv/bin/python edge_nodes.py
```

Use the same width-profile color grayscale, with the measured 7.83 px FWHM.
For top/bottom borders, average only the first/last 24 real image rows with
inward-decaying weights, then apply a 1D vertical-string cross-section filter
along x. For left/right borders, use the corresponding columns and filter along
y for horizontal strings. The positive Gaussian has the measured string FWHM;
negative flanks estimate local background. Kernel weights are renormalized at
corners using real pixels, rather than inventing content outside the image.

Apply the existing shallow-slope threshold criterion to the pooled four edge
score profiles. Each connected above-threshold 1D interval gives a geometric
centroid, projected exactly onto its image border. These are proposals only:
no mass cutoff, de-duplication, or final unwanted-node filtering is applied.
Corner proposals from different sides remain separate if they occur.

`output/edge_nodes/edge_heatmap.png` shows the 1D edge scores in **5-pixel border
bands for visibility**, not a full 2D interior score map. `edge_nodes.png` shows
above-threshold edge scores and yellow endpoint proposals, alongside the 80
existing white crossing nodes. `straight_grid.png` draws vertical/horizontal
straight lines at all cleaned x/y maxima (13 and 8), with the same two node sets.
It does not refit spacing/phase, use the Bezier curves, snap nodes to the grid,
or apply any additional filtering.

The current shared edge threshold is 0.2641, yielding 27 proposals: top 8,
bottom 3, left 8, right 8. This is conservative, especially at the bottom.
`--threshold` overrides it for inspection. `edge_profiles.png`, per-edge CSVs,
and `nodes.json` retain scores, automatic threshold parameters, endpoint
coordinates, and grid coordinates; PNG/PDF plots are provided.

## Complete edge endpoints before maxima cleanup

```sh
.venv/bin/python complete_edge_nodes.py
.venv/bin/python scan_node_lines.py --nodes output/completed_edge_nodes/nodes.json --output output/completed_edge_nodes/scans
.venv/bin/python clean_scan_peaks.py --scans output/completed_edge_nodes/scans --output output/completed_edge_nodes/cleaned
```

Preserve all 27 accepted edge proposals at their measured positions. Match them
one-to-one to the existing straight grid's line endpoints within half the median
spacing. For each missing endpoint, use the nearest local maximum in its line's
midpoint-bounded search interval, from either the edge score or the crossing
score. The latter is projected onto the border by taking its maximum within the
same 24-pixel inward strip. Additional candidates need only have positive score;
no old detection threshold is imposed. Source, score and distance are recorded.
No suitable local maximum raises an error instead of inventing a node.

This uses the preceding straight grid as a bootstrap, then adds the endpoint
observations **before** rerunning scans and maxima cleanup. The current run adds
15 endpoints (8 from edge scores and 7 from cross scores), giving exactly 13 top,
13 bottom, 8 left and 8 right endpoints. The 84 pre-cleanup crossing nodes plus
42 endpoints make 126 nodes. Endpoints bypass crossing-region mass filtering.

Each endpoint carries its known family: top/bottom nodes vote in the x-scan,
left/right nodes in the y-scan. Interior crossing nodes vote in both. This avoids
creating artificial strings from image-border coordinate pileups. The x-scan
therefore has 110 contributing nodes; the initial y-scan has 100.

X-cleanup removes the same 98 and 202 px peaks and four interior nodes, preserving
all 42 endpoints. The y-scan is then recomputed from 96 eligible nodes (80 interior
plus 16 left/right endpoints); its rejected peaks are 153 and 244 px. Y cleanup
remains a peak-only diagnostic and does not delete more 2D nodes. The total
retained node set has 122 entries. No further unwanted-node filtering is applied.

`output/completed_edge_nodes/cleaned/cleaned_scans.png`/`.pdf` shows retained and
removed maxima and before/after local spacing in both directions. JSON/CSV
exports retain endpoint lineage, assignments, raw scans and cleanup decisions.
All previous result folders are preserved.

## General line-support distribution after maxima cleanup

```sh
.venv/bin/python node_line_support.py
```

The current inspection stage uses **all 122 nodes after maxima filtering**,
restoring the node previously removed by the isolated-neighbor experiment. The
four interior nodes rejected by maxima filtering remain excluded. This does not
apply the isolated-node filter.

At each node, evaluate the same Gaussian proximity score on the vertical and
horizontal line through it, with the existing scan tolerance. A node does not
support itself. Only direction-eligible nodes contribute; top/bottom endpoints
belong to vertical-line support, left/right to horizontal-line support. Average
the eligible directional scores per node. This is a general score computed from
all nodes, with no coordinate- or node-ID-specific exception.

`output/node_line_support/line_score_distribution.png`/`.pdf` shows 20-bin
histograms for each direction and the combined average, with mean and population
standard deviation. `scores.csv` and `analysis.json` export every node's scores,
statistics and whether the average falls below mean-SD. **No removal is applied.**
The exported analysis retains all nodes and can be used for subsequent fitting.

For this image, average line support has mean 5.920 and SD 2.081, giving lower
cutoff 3.839. Seventeen nodes fall below it, including the previously isolated
node (average support 1.319). Thus this criterion is not specific to that node and
would also flag 16 others; the distribution is provided for judging that choice.
Raw support depends on how many detections are available in each direction, so
sparsely sampled legitimate strings and edges can also get low scores.

## Within-maximum coordinate deviations

```sh
.venv/bin/python local_node_deviations.py
```

Start with the same 122 nodes after maxima cleanup, without applying the line-score
or isolated-node filters. Recompute each directional line scan from its eligible
nodes, then find **all local minima of that actual signal**. Consecutive minima
bound coordinate groups; the image edges close the outer groups. Intervals are
left-inclusive/right-exclusive, so a node exactly at a minimum belongs to the
right-hand group.

For each group independently, compute the population mean and standard deviation
of its node **coordinates**, including each node itself. Report signed and
absolute `(coordinate - group_mean) / group_SD`. This measures position scatter
within a scan peak, not deviation of the node's line-score value. Zero-variance
groups have undefined z-scores, explicitly exported as null rather than zero.
No node removal or rejection threshold is applied.

`output/local_node_deviations/local_deviations.png`/`.pdf` shows coordinate versus
absolute local z-score (with minima as vertical boundaries), plus the original
image with nodes colored by their local x/y deviations. CSV and JSON contain
per-node means, SDs, group membership and z-scores. There are 13 x-groups (106
eligible nodes) and 10 y-groups (96 eligible nodes). All minima are used literally,
including those surrounding the y-scan's secondary peaks; the earlier retained
maximum list is not used to merge these groups.
