"""Conservative tracking of an approved node roster before curve fitting.

Missing observations are ordinary data, not invented measurements. Unmatched
detections never create verified tracks automatically. All gates are heuristic.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from session_data import validate_nodes

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
TRACKING_POLICY_VERSION = 3


@dataclass(frozen=True)
class TrackingConfig:
    use_flow: bool = True
    use_velocity: bool = True
    max_missing: int = 3
    gate_fraction: float = .40
    min_correlation: float = .60
    recovery_correlation: float = .78
    ambiguity_margin: float = .12
    candidate_lifetime: int = 2

    def __post_init__(self):
        if self.max_missing < 0 or self.candidate_lifetime < 0 or not 0 < self.gate_fraction < .5:
            raise ValueError('Require nonnegative missing limits and a gate below half the grid spacing')
        if not -1 < self.min_correlation <= self.recovery_correlation < 1 or self.ambiguity_margin < 0:
            raise ValueError('Invalid correlation or ambiguity thresholds')


def patch(image, xy, radius):
    return cv2.getRectSubPix(image, (2*radius+1, 2*radius+1), tuple(map(float, xy)))


def correlation(a, b):
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    a, b = a-a.mean(), b-b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator > 1e-8 else -1.


def assign(cost, unmatched=1.):
    """One-to-one matching, with a separate finite unmatched choice per track."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or np.isnan(cost).any():
        raise ValueError('Assignment needs a 2D cost matrix without NaNs')
    count, columns = cost.shape
    if not count:
        return {}
    padded = np.column_stack((cost, np.full((count, count), unmatched)))
    rows, indices = linear_sum_assignment(padded)
    return {int(i): int(j) for i, j in zip(rows, indices) if j < columns and cost[i, j] < unmatched}


def compatible(track, node):
    return track['kind'] == node['kind'] and (track['kind'] != 'edge' or track['side'] == node['side'])


def topology_conflicts(nodes, proposals, width):
    """Reject observations that reverse verified grid order or merge crossings."""
    conflicts = set()
    indices = sorted(proposals)
    for offset, i in enumerate(indices):
        a = nodes[i]
        for j in indices[offset+1:]:
            b = nodes[j]
            p, q = proposals[i]['xy'], proposals[j]['xy']
            if a['kind'] == b['kind'] == 'interior' and np.linalg.norm(p-q) < 1.5*width:
                conflicts.update((i, j))
            for label, order, axis in (('row', 'column', 0), ('column', 'row', 1)):
                if a[label] is not None and a[label] == b[label] and a[order] is not None and b[order] is not None:
                    coordinate = 'x' if axis == 0 else 'y'
                    if (a[coordinate]-b[coordinate])*(p[axis]-q[axis]) <= 0:
                        conflicts.update((i, j))
    return conflicts


class NodeTracker:
    def __init__(self, nodes, image, width, spacing, dt=1., config=None):
        self.config = config or TrackingConfig()
        self.width, self.spacing, self.dt = float(width), float(spacing), float(dt)
        if min(self.width, self.spacing, self.dt) <= 0 or not np.isfinite([width, spacing, dt]).all():
            raise ValueError('Width, grid spacing, and frame interval must be finite and positive')
        self.size = np.array(image.shape[::-1])
        if image.dtype != np.uint8 or image.ndim != 2 or not nodes:
            raise ValueError('Require verified nodes and a uint8 grayscale reference image')
        validate_nodes(nodes, self.size.tolist(), require_crossing=False)
        self.radius = max(4, int(np.ceil(1.5*width)))
        self.gate = min(3*width, self.config.gate_fraction*spacing)
        self.nodes = list(nodes)
        self.frame, self.time, self.previous_image = 0, 0., image
        self.tracks = [self.new_track(n, image) for n in nodes]
        self.candidates, self.next_candidate, self.candidate_events = [], 0, []

    def new_track(self, node, image):
        xy = np.array([node['x'], node['y']], dtype=float)
        return dict(mean=np.r_[xy, 0., 0.], covariance=np.diag([(self.width/3)**2]*2+[(self.width/self.dt)**2]*2),
                    missed=0, observations=1, last_xy=xy, last_seen=self.frame, last_time=self.time,
                    template=patch(image, xy, self.radius), velocity=None)

    def initial(self, nodes=None):
        return [dict(n, status='verified' if n.get('human_edited', True) else 'observed',
                     measurement_source='human' if n.get('human_edited', True) else 'independent_detection',
                     measured_xy=[n['x'], n['y']],
                     filtered_xy=[n['x'], n['y']], prediction_xy=None, missed=0, velocity=None,
                     acceleration=None, measurement_velocity=None, position_sd=self.width/3,
                     reason=None, detection_id=n['id'], correlation=None, forward_backward_error=None,
                     last_observed_frame=self.frame, observations=1) for n in (self.nodes if nodes is None else nodes)]

    def constrain(self, edits, image, records):
        """Update just the edited tracks; other observations keep their own evidence."""
        records = {n['id']: n for n in records}
        for key, node in edits.items():
            position = next((i for i, n in enumerate(self.nodes) if n['id'] == key), None)
            if node is None:
                records.pop(key, None)
                if position is not None:
                    track = self.tracks[position]
                    track['missed'] = self.config.max_missing+1
                    track['velocity'] = None
                    track['mean'][2:] = 0
                continue
            validate_nodes([node], self.size.tolist(), require_crossing=False)
            node = dict(node, human_edited=True)
            if position is None:
                self.nodes.append(node)
                self.tracks.append(self.new_track(node, image))
            else:
                self.nodes[position] = node
                self.tracks[position] = self.new_track(node, image)
            records[key] = self.initial([node])[0]
        return list(records.values())

    def anchor(self, nodes, image, dt=None):
        """Apply exact human positions; absent identities remain recoverable."""
        validate_nodes(nodes, self.size.tolist(), require_crossing=False)
        if image.shape != self.previous_image.shape or image.dtype != np.uint8:
            raise ValueError('Anchor dimensions or image type changed')
        self.frame += 1
        self.time += self.step_interval(dt)
        visible = {n['id']: n for n in nodes}
        for i, node in enumerate(self.nodes):
            if node['id'] in visible:
                self.nodes[i] = visible[node['id']]
                self.tracks[i] = self.new_track(visible[node['id']], image)
            else:
                self.tracks[i]['missed'] = self.config.max_missing+1
                self.tracks[i]['mean'][:2] = self.tracks[i]['last_xy']
                self.tracks[i]['mean'][2:] = 0
                self.tracks[i]['velocity'] = None
        known = {n['id'] for n in self.nodes}
        for node in nodes:
            if node['id'] not in known:
                self.nodes.append(node)
                self.tracks.append(self.new_track(node, image))
        self.previous_image = image
        self.candidate_events = []
        # New anchors replace appearance templates, but never move their nodes.
        return self.initial(nodes)

    def flow(self, image, predictions):
        indices = [i for i, (n, t) in enumerate(zip(self.nodes, self.tracks))
                   if n['kind'] == 'interior' and t['missed'] == 0]
        if not indices:
            return {}
        previous = np.array([self.tracks[i]['last_xy'] for i in indices], dtype=np.float32).reshape(-1, 1, 2)
        initial = np.array([predictions[i] for i in indices], dtype=np.float32).reshape(-1, 1, 2)
        window = max(15, 2*int(np.ceil(2*self.width))+1)
        params = dict(winSize=(window, window), maxLevel=2,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        current, forward, error = cv2.calcOpticalFlowPyrLK(self.previous_image, image, previous, initial,
                                                          flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **params)
        if current is None:
            return {}
        # Replace failed output coordinates before passing them back into OpenCV.
        valid = forward.ravel().astype(bool) & np.isfinite(current).all(axis=(1, 2))
        current[~valid] = previous[~valid]
        reverse, backward, _ = cv2.calcOpticalFlowPyrLK(image, self.previous_image, current, previous.copy(),
                                                       flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **params)
        if reverse is None:
            return {}
        result = {}
        for j, i in enumerate(indices):
            xy = current[j, 0].astype(float)
            fb = float(np.linalg.norm(reverse[j, 0]-previous[j, 0]))
            ncc = correlation(self.tracks[i]['template'], patch(image, xy, self.radius)) if np.isfinite(xy).all() else -1
            if (valid[j] and backward[j, 0] and np.isfinite(fb) and fb <= max(1., .25*self.width)
                    and error[j, 0] < 25 and self.inside(xy) and np.linalg.norm(xy-predictions[i]) < self.gate
                    and ncc >= self.config.min_correlation):
                result[i] = dict(xy=xy, correlation=ncc, forward_backward_error=fb,
                                 source='optical_flow', detection_id=None)
        return result

    def inside(self, xy):
        return bool(np.isfinite(xy).all() and np.all(xy >= 0) and np.all(xy <= self.size-1))

    def step_interval(self, dt):
        value = self.dt if dt is None else float(dt)
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Frame interval must be positive and finite')
        return value

    def step(self, image, detections, dt=None):
        if image.dtype != np.uint8 or image.shape != self.previous_image.shape:
            raise ValueError('Tracking requires equally sized uint8 grayscale frames')
        if len({d['id'] for d in detections}) != len(detections):
            raise ValueError('Duplicate detection IDs within one frame')
        for node in detections:
            if node['kind'] not in ('interior', 'edge'):
                raise ValueError(f"Unknown detection kind {node['kind']}")
            if not self.inside(np.array([node['x'], node['y']])):
                raise ValueError(f"Invalid detection coordinates for ID {node['id']}")
        dt = self.step_interval(dt)
        self.frame += 1
        self.time += dt
        transition = np.eye(4)
        transition[:2, 2:] = dt*np.eye(2)
        noise = np.vstack((.5*dt**2*np.eye(2), dt*np.eye(2)))
        process_covariance = (.20*self.width/self.dt**2)**2*(noise@noise.T)
        for t in self.tracks:
            t['mean'] = transition@t['mean']
            t['covariance'] = transition@t['covariance']@transition.T+process_covariance
        for node, t in zip(self.nodes, self.tracks):
            if not self.config.use_velocity or t['missed'] or node['kind'] == 'edge':
                t['mean'][:2] = t['last_xy']
                if t['missed'] or node['kind'] == 'edge':
                    t['mean'][2:] = 0
        predictions = np.array([t['mean'][:2].copy() for t in self.tracks])
        gates = [min(self.config.gate_fraction*self.spacing,
                     self.gate+2*np.sqrt(np.max(np.diag(t['covariance'])[:2]))) for t in self.tracks]
        flows = self.flow(image, predictions) if self.config.use_flow else {}
        cost = np.full((len(self.tracks), len(detections)), np.inf)
        appearance = np.full_like(cost, -1.)
        patches = [patch(image, [n['x'], n['y']], self.radius) for n in detections]
        for i, (node, track) in enumerate(zip(self.nodes, self.tracks)):
            center = flows[i]['xy'] if i in flows else predictions[i]
            for j, detected in enumerate(detections):
                xy = np.array([detected['x'], detected['y']])
                distance = np.linalg.norm(xy-predictions[i])
                if not compatible(node, detected) or distance > gates[i]:
                    continue
                if i in flows and np.linalg.norm(xy-center) > 1.5*self.width:
                    continue
                ncc = correlation(track['template'], patches[j])
                appearance[i, j] = ncc
                strict = track['missed'] > 0 or not detected.get('spatial_accepted', True)
                minimum = self.config.recovery_correlation if strict else self.config.min_correlation
                if ncc < minimum:
                    continue
                cost[i, j] = .65*np.linalg.norm(xy-center)/gates[i]+.35*(1-ncc)/2
                if not detected.get('spatial_accepted', True):
                    cost[i, j] += .10
        matches = assign(cost)
        proposals, reasons = {}, {}
        for i, track in enumerate(self.tracks):
            if i in matches:
                j = matches[i]
                alternatives = np.sort(cost[i, np.isfinite(cost[i])])
                if cost[i, j] > alternatives[0]+self.config.ambiguity_margin:
                    reasons[i] = 'assignment_conflict'
                elif len(alternatives) > 1 and alternatives[1]-alternatives[0] < self.config.ambiguity_margin:
                    reasons[i] = 'ambiguous_detections'
                else:
                    proposals[i] = dict(xy=np.array([detections[j]['x'], detections[j]['y']]),
                                        source='detection', detection_id=detections[j]['id'],
                                        correlation=float(appearance[i, j]), forward_backward_error=None)
            elif i in flows:
                proposals[i] = flows[i]
            else:
                reasons[i] = 'no_reliable_observation'
        for i in topology_conflicts(self.nodes, proposals, self.width):
            proposals.pop(i)
            reasons[i] = 'grid_identity_conflict'
        # Prevent an observation from silently entering another track's search
        # region while that track is missing: repeating crossings look alike.
        for i in list(proposals):
            for j, node in enumerate(self.nodes):
                if j != i and j not in proposals and compatible(self.nodes[i], node):
                    if np.linalg.norm(proposals[i]['xy']-predictions[j]) < 1.5*self.width:
                        proposals.pop(i)
                        reasons[i] = 'unresolved_neighbor'
                        break
        records, used = [], set()
        for i, (node, track) in enumerate(zip(self.nodes, self.tracks)):
            observation = proposals.get(i)
            old_velocity = track['velocity']
            acceleration, measurement_velocity = None, None
            if observation is not None:
                xy = observation['xy']
                measurement_velocity = ((xy-track['last_xy'])/(self.time-track['last_time'])).tolist()
                variance = (self.width/(4 if observation['source'] == 'detection' else 3))**2
                covariance = track['covariance']
                gain = np.linalg.solve(covariance[:2, :2]+variance*np.eye(2), covariance[:2, :]).T
                track['mean'] += gain@(xy-track['mean'][:2])
                update = np.eye(4)-gain@np.eye(2, 4)
                track['covariance'] = update@covariance@update.T+variance*gain@gain.T
                track['velocity'] = track['mean'][2:].copy()
                if old_velocity is not None and track['missed'] == 0:
                    acceleration = ((track['velocity']-old_velocity)/dt).tolist()
                status = 'recovered' if track['missed'] else 'observed'
                track.update(missed=0, last_xy=xy.copy(), last_seen=self.frame, last_time=self.time,
                             observations=track['observations']+1, template=patch(image, xy, self.radius))
                if observation['detection_id'] is not None:
                    used.add(observation['detection_id'])
            else:
                track['missed'] += 1
                xy = track['last_xy'].copy()
                track['mean'][:2], track['mean'][2:] = xy, 0
                status = 'held' if node['kind'] == 'edge' else 'lost' if track['missed'] > self.config.max_missing else 'predicted'
                if node['kind'] == 'edge':
                    reasons[i] = 'edge_not_identified_position_held'
                track['velocity'] = None
            record = dict(node, x=float(xy[0]), y=float(xy[1]), status=status, missed=track['missed'], human_edited=False,
                          measured_xy=xy.tolist() if observation else None,
                          prediction_xy=predictions[i].tolist(), filtered_xy=track['mean'][:2].tolist(),
                          position_sd=float(np.sqrt(np.max(np.diag(track['covariance'])[:2]))), search_radius=gates[i],
                          velocity=track['velocity'].tolist() if track['velocity'] is not None else None,
                          acceleration=acceleration, measurement_velocity=measurement_velocity,
                          measurement_source=observation['source'] if observation else None,
                          detection_id=observation['detection_id'] if observation else None,
                          correlation=observation['correlation'] if observation else None,
                          forward_backward_error=observation['forward_backward_error'] if observation else None,
                          reason=reasons.get(i), last_observed_frame=track['last_seen'], observations=track['observations'])
            if node['kind'] == 'edge':
                # Boundary intersections are not material points.
                record.update(velocity=None, acceleration=None, measurement_velocity=None)
            records.append(record)
        unclaimed = self.update_candidates(image, detections, used, records)
        self.previous_image = image
        return records, unclaimed

    def update_candidates(self, image, detections, used, records):
        self.candidate_events = []
        pending, reports = [], []
        for d in detections:
            if d['id'] in used:
                continue
            xy = np.array([d['x'], d['y']])
            neighbors = [r for r in records if compatible(r, d) and np.linalg.norm(xy-[r['x'], r['y']]) <= r['search_radius']]
            if neighbors:
                reason = 'near_existing_track'
            elif not d.get('spatial_accepted', True):
                reason = 'spatially_rejected'
            else:
                reason = 'unverified_new'
                pending.append(d)
            reports.append(dict(d, reason=reason, candidate_id=None))
        existing = [c for c in self.candidates if self.frame-c['last_seen'] <= self.config.candidate_lifetime+1]
        cost = np.full((len(existing), len(pending)), np.inf)
        for i, c in enumerate(existing):
            for j, d in enumerate(pending):
                distance = np.linalg.norm(np.array([d['x'], d['y']])-c['xy'])
                if compatible(c, d) and distance < self.gate:
                    if correlation(c['template'], patch(image, [d['x'], d['y']], self.radius)) >= self.config.min_correlation:
                        cost[i, j] = distance/self.gate
        matches = assign(cost)
        matched_columns = set(matches.values())
        for i, j in matches.items():
            c, d = existing[i], pending[j]
            c.update(xy=np.array([d['x'], d['y']]), last_seen=self.frame, hits=c['hits']+1)
            d['_candidate_id'] = c['candidate_id']
        for j, d in enumerate(pending):
            if j not in matched_columns:
                c = dict(candidate_id=f'candidate-{self.next_candidate}', xy=np.array([d['x'], d['y']]),
                         kind=d['kind'], side=d.get('side'), first_seen=self.frame, last_seen=self.frame, hits=1,
                         template=patch(image, [d['x'], d['y']], self.radius))
                self.next_candidate += 1
                existing.append(c)
                d['_candidate_id'] = c['candidate_id']
                self.candidate_events.append(dict(candidate_id=c['candidate_id'], event='created'))
        identities = {d['id']: d.pop('_candidate_id') for d in pending}
        for r in reports:
            r['candidate_id'] = identities.get(r['id'])
            if r['candidate_id'] is not None:
                c = next(c for c in existing if c['candidate_id'] == r['candidate_id'])
                r['candidate_hits'] = c['hits']
                r['candidate_first_seen'] = c['first_seen']
        for c in self.candidates:
            if self.frame-c['last_seen'] > self.config.candidate_lifetime:
                self.candidate_events.append(dict(candidate_id=c['candidate_id'], event='expired'))
        self.candidates = [c for c in existing if self.frame-c['last_seen'] <= self.config.candidate_lifetime]
        return reports

    def candidate_states(self):
        return [dict(candidate_id=c['candidate_id'], xy=c['xy'].tolist(), kind=c['kind'], side=c.get('side'),
                     first_seen=c['first_seen'], last_seen=c['last_seen'], hits=c['hits'],
                     status='tentative' if c['last_seen'] == self.frame else 'missing') for c in self.candidates]


def reconcile_nodes(forward, backward, definitions, width):
    """Use the better supported direction; disagreements require human review."""
    fwd, back = ({n['id']: n for n in nodes} for nodes in (forward, backward))
    result = []
    for key in sorted(fwd.keys() | back.keys()):
        a, b = fwd.get(key), back.get(key)
        observed = [n for n in (a, b) if n is not None and n['measured_xy'] is not None]
        if not observed:
            result.append(dict(a if a is not None else b))
            continue
        best = max(observed, key=lambda n: (n.get('correlation') or 0)+(.02 if n['measurement_source'] == 'detection' else 0))
        manual = [n for n in observed if n.get('human_edited')]
        if manual:
            result.append(dict(manual[0]))
            continue
        chosen = dict(best, temporal_direction='forward' if best is a else 'backward')
        if len(observed) == 2 and np.linalg.norm(np.array(a['measured_xy'])-b['measured_xy']) > 2*width:
            chosen.update(status='predicted', measured_xy=None, velocity=None, acceleration=None,
                          measurement_source=None, reason='forward_backward_disagreement')
        result.append(chosen)
    base = [definitions[n['id']] for n in result]
    proposals = {i: dict(xy=np.array(n['measured_xy'])) for i, n in enumerate(result) if n['measured_xy'] is not None}
    for i in topology_conflicts(base, proposals, width):
        if result[i].get('human_edited'):
            continue
        result[i].update(status='predicted', measured_xy=None, velocity=None, acceleration=None,
                         measurement_source=None, reason='grid_identity_conflict')
    return result


def hold_missing_positions(frames):
    """Apply the same last-observation rule after bidirectional reconciliation."""
    last, deleted = {}, set()
    for frame in frames:
        for key in frame.get('deleted_node_ids', []):
            last.pop(key, None)
            deleted.add(key)
        kept = []
        for node in frame['nodes']:
            if node.get('measured_xy') is not None:
                deleted.discard(node['id'])
                last[node['id']] = list(node['measured_xy'])
            elif node['id'] in deleted:
                continue
            elif node['id'] in last:
                node['x'], node['y'] = last[node['id']]
                node['filtered_xy'] = list(last[node['id']])
                node['velocity'], node['acceleration'] = None, None
                if node['kind'] == 'edge' and node.get('status') != 'human_absent':
                    node.update(status='held', reason='edge_not_identified_position_held')
            kept.append(node)
        frame['nodes'] = kept
