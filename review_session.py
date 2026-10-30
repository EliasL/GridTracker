"""Persistent human anchors, reanalysis, and exports with explicit estimates."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import time

import numpy as np

from mesh_hints import identity
from session_data import validate_nodes, parse_fps, validate_reference, write_json, load_draft, node_geometry, label_node

_SESSION_FPS = object()


def validate_roster(anchors):
    ids, identities = {}, {}
    for document in anchors.values():
        for node in document['nodes']:
            key = identity(node)
            if node['id'] in ids and ids[node['id']] != key:
                raise ValueError(f"Node ID {node['id']} changes string identity between confirmed frames")
            if key in identities and identities[key] != node['id']:
                raise ValueError(f'Grid identity {key} has different IDs between confirmed frames; restore the existing ID')
            ids[node['id']], identities[key] = key, node['id']


def update_kinematics(report, include_estimates=False):
    """Differentiate contiguous measurements only; never bridge an occlusion."""
    dt = 1/report['fps'] if report.get('fps') is not None else 1.
    frames = report['frames']
    times = np.array([f.get('frame_offset', f['frame'])*dt for f in frames])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Frame times must be finite and strictly increasing')
    by_frame = [{n['id']: n for n in f['nodes']} for f in frames]
    for frame in frames:
        frame['time'] = frame.get('frame_offset', frame['frame'])*dt
        for node in frame['nodes']:
            node['velocity'], node['acceleration'] = None, None
    for key in {n['id'] for f in frames for n in f['nodes']}:
        segment, segment_times = [], []
        for index in range(len(frames)+1):
            node = by_frame[index].get(key) if index < len(frames) else None
            xy = node.get('measured_xy') if node is not None else None
            if node is not None and include_estimates and xy is None:
                xy = [node.get('x'), node.get('y')]
                if any(v is None for v in xy) or not np.isfinite(xy).all():
                    xy = None
            if node is not None and node['kind'] == 'interior' and xy is not None:
                segment.append(node)
                segment_times.append(times[index])
                continue
            if len(segment) >= 2:
                xy = np.array([n['measured_xy'] if n.get('measured_xy') is not None else [n['x'], n['y']] for n in segment])
                velocity = np.gradient(xy, segment_times, axis=0, edge_order=2 if len(segment) >= 3 else 1)
                acceleration = np.gradient(velocity, segment_times, axis=0, edge_order=2) if len(segment) >= 3 else None
                for j, value in enumerate(segment):
                    value['velocity'] = velocity[j].tolist()
                    value['acceleration'] = acceleration[j].tolist() if acceleration is not None else None
                    value['kinematics_uses_estimates'] = any(n.get('measured_xy') is None for n in segment)
            segment, segment_times = [], []
    report.update(time_unit='seconds' if report.get('fps') is not None else 'frames',
                  velocity_unit='pixels/second' if report.get('fps') is not None else 'pixels/frame',
                  acceleration_unit='pixels/second^2' if report.get('fps') is not None else 'pixels/frame^2',
                  kinematics_method='finite differences of contiguous measured image positions; second-order endpoints when possible')


class ReviewSession:
    def __init__(self, results, reference, output, report=None, fps=None, method='flow'):
        self.results, self.output, self.method = results, Path(output), method
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output/'review.json'
        from calibration import load_calibration
        self.calibration = load_calibration(self.output)
        self.anchors = {0: deepcopy(reference)}
        self.fps = parse_fps(fps if fps is not None else reference.get('fps'))
        self.dirty = False
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state['source_sha256'] != [r['source_sha256'] for r in results]:
                raise ValueError('Saved review belongs to different input frames')
            self.anchors = {int(k): v for k, v in state['anchors'].items()}
            self.fps, self.dirty = parse_fps(state['fps']), state['dirty']
            if self.anchors[0] != reference:
                raise ValueError('Saved first-frame review differs from reference.json; use the matching reference')
        for index, document in self.anchors.items():
            if not 0 <= index < len(results):
                raise ValueError(f'Confirmed frame {index} is outside this sequence')
            validate_reference(document, results[index])
        validate_roster(self.anchors)
        if fps is not None and self.fps != parse_fps(fps):
            self.fps, self.dirty = parse_fps(fps), True
        self.report = deepcopy(report) if report is not None else None
        if self.report is not None:
            self.report['fps'] = self.fps
            for frame in self.report['frames']:
                frame['human_verified'] = frame['frame'] in self.anchors
                frame['width'] = results[frame['frame']]['width']
            update_kinematics(self.report)
        self.save()

    def save(self):
        from calibration import load_calibration
        self.calibration = load_calibration(self.output)
        write_json(self.path, dict(source_sha256=[r['source_sha256'] for r in self.results],
                                   anchors={str(k): v for k, v in self.anchors.items()},
                                   fps=self.fps, dirty=self.dirty, method=self.method))
        if self.report is not None:
            self.report['confirmed_frames'] = sorted(self.anchors)
            self.report['needs_reanalysis'] = self.dirty
            self.report['calibration'] = self.calibration
            write_json(self.output/'tracking.json', self.report)

    def roster(self):
        known = {}
        if self.report is not None:
            known.update({n['id']: n for f in self.report['frames'] for n in f['nodes']})
        known.update({n['id']: n for d in self.anchors.values() for n in d['nodes']})
        return list(known.values())

    def validate_edit(self, index, nodes):
        validate_nodes(nodes, self.results[index]['image_size'], require_crossing=index == 0)
        prospective = {**self.anchors, index: dict(nodes=nodes)}
        validate_roster(prospective)

    def confirm(self, index, document):
        from fit_bezier_curves import fit_labeled_curves
        validate_reference(document, self.results[index])
        self.validate_edit(index, document['nodes'])
        self.anchors[index] = deepcopy(document)
        self.fps, self.dirty = parse_fps(document.get('fps')), True
        if index == 0:
            write_json(self.output/'reference.json', document)
        if self.report is not None:
            # The editor's exact coordinates define the frame. No tracker is run.
            frame = self.report['frames'][index]
            frame['nodes'] = [dict(n, measured_xy=[n['x'], n['y']], status='verified',
                                    measurement_source='human', reason=None, velocity=None, acceleration=None,
                                    prediction_xy=None, filtered_xy=[n['x'], n['y']], position_sd=None,
                                    missed=0, correlation=None) for n in document['nodes']]
            frame['human_verified'] = True
            frame['curves'], frame['ignored_curves'] = fit_labeled_curves(frame['nodes'], frame['image_size'], self.results[index]['width'])
            frame['unclaimed_detections'], frame['candidate_tracks'], frame['candidate_events'] = [], [], []
            frame['counts'] = {status: len(frame['nodes']) if status == 'verified' else 0 for status in frame['counts']}
            self.report['fps'] = self.fps
            update_kinematics(self.report)
        self.save()

    def unmark(self, index):
        if index == 0:
            raise ValueError('The first frame remains the human reference; edit its nodes and confirm to correct it')
        if index not in self.anchors:
            return False
        history = self.output/'reviews'/'history'
        history.mkdir(parents=True, exist_ok=True)
        stamp = time.time_ns()
        write_json(history/f'frame_{index:02d}_{stamp}_anchor.json', self.anchors[index])
        for suffix in ('.json', '.draft.json'):
            path = self.output/'reviews'/f'frame_{index:02d}{suffix}'
            if path.exists():
                path.replace(history/f'{stamp}_{path.name}')
        del self.anchors[index]
        self.dirty = True
        if self.report is not None:
            frame = self.report['frames'][index]
            frame['human_verified'] = False
            for node in frame['nodes']:
                node.update(status='held' if node['kind'] == 'edge' else 'predicted', measured_xy=None,
                            measurement_source=None, velocity=None, acceleration=None, reason='frame_unmarked_pending_reanalysis')
            frame['counts'] = {status: sum(n['status'] == status for n in frame['nodes']) for status in frame['counts']}
            update_kinematics(self.report)
        self.save()
        return True

    def reanalyze(self, progress=None, frame_range=None):
        from run_temporal import propagate
        snapshots = {i: deepcopy(d['nodes']) for i, d in self.anchors.items()}
        report = propagate(self.results, self.anchors[0], self.output, self.fps, self.method, confirmed=self.anchors,
                           progress=progress, frame_range=frame_range, previous_report=self.report)
        for index, expected in snapshots.items():
            actual = [{k: n[k] for k in ('id', 'x', 'y', 'row', 'column', 'kind')} for n in report['frames'][index]['nodes']]
            frozen = [{k: n[k] for k in ('id', 'x', 'y', 'row', 'column', 'kind')} for n in expected]
            if actual != frozen:
                raise RuntimeError(f'Reanalysis changed confirmed frame {index}')
        self.report, self.dirty = report, self.dirty if frame_range is not None else False
        self.save()
        return report

    def saved_drafts(self):
        from fit_bezier_curves import node_available
        drafts = {}
        for result in self.results:
            index = result['frame']
            path = self.output/'reviews'/f'frame_{index:02d}.draft.json'
            if path.exists():
                document = load_draft(path, result)
                original = self.report['frames'][index]['nodes']
                if document.get('editor_state', {}).get('modified', node_geometry(document['nodes']) != node_geometry(original)):
                    drafts[index] = [n for n in document['nodes'] if node_available(n)]
        return drafts

    def export_csv(self, path=None, draft=None, fps=_SESSION_FPS, drafts=None):
        if self.report is None:
            self.reanalyze()
        report = deepcopy(self.report)
        report['fps'] = self.fps if fps is _SESSION_FPS else parse_fps(fps)
        report['needs_reanalysis'] = self.dirty
        drafts = dict(drafts or {})
        if draft is not None:
            drafts[draft[0]] = draft[1]
        for index, nodes in drafts.items():
            if len({n['id'] for n in nodes}) != len(nodes):
                raise ValueError('Correct duplicate node IDs in the draft before export')
            frame = report['frames'][index]
            frame['human_verified'] = False
            frame['nodes'] = [dict(n, measured_xy=[n['x'], n['y']], status='draft', measurement_source='human_draft') for n in nodes]
            from fit_bezier_curves import fit_labeled_curves
            frame['curves'], frame['ignored_curves'] = fit_labeled_curves(frame['nodes'], frame['image_size'], self.results[index]['width'])
        return export_report(report, self.results, self.output, path)


def curve_crossing_snapshot(report):
    """Export geometry is derived from the displayed curve fits, never guide points."""
    from fit_bezier_curves import curve_intersections
    report = deepcopy(report)
    families = {'horizontal': 'row', 'vertical': 'column'}
    global_labels = {label: set() for label in families.values()}
    for frame in report['frames']:
        for node in frame['nodes']:
            for label in global_labels:
                if node.get(label) is not None:
                    global_labels[label].add(node[label])
        for curve in frame['curves']+frame['ignored_curves']:
            global_labels[families[curve['family']]].add(curve['line_id'])
    for frame in report['frames']:
        fits = {family: {} for family in families}
        for curve in frame['curves']:
            group = fits[curve['family']]
            if curve['line_id'] in group:
                raise ValueError('Duplicate fitted string identity')
            group[curve['line_id']] = curve
        labels = global_labels if report.get('identities_linked', True) else {
            label: {n[label] for n in frame['nodes'] if n.get(label) is not None}
            for label in global_labels}
        guides = frame['nodes']
        nodes = []
        for row in sorted(labels['row']):
            for column in sorted(labels['column']):
                vertical, horizontal = fits['vertical'].get(column), fits['horizontal'].get(row)
                points = curve_intersections(vertical, horizontal, frame['image_size']) if vertical is not None and horizontal is not None else np.empty((0, 2))
                count = len(points)
                xy = points[0].tolist() if count == 1 else [None, None]
                status = ('missing_curve' if vertical is None or horizontal is None else
                          'no_intersection' if count == 0 else 'ambiguous_intersections' if count > 1 else 'curve_crossing')
                nodes.append(dict(id=f'r{row}:c{column}', row=row, column=column, kind='interior',
                                  x=xy[0], y=xy[1], measured_xy=None, status=status,
                                  measurement_source='bezier_intersection', estimate_source='bezier_intersection',
                                  position_verified=False, intersection_count=count, intersections=points.tolist(),
                                  guide_node_ids=[n['id'] for n in guides if n['kind'] == 'interior' and n['row'] == row and n['column'] == column],
                                  velocity=None, acceleration=None))
        frame['nodes'] = nodes
    if report.get('identities_linked', True):
        update_kinematics(report, include_estimates=True)
        report['kinematics_method'] = 'finite differences of contiguous Bezier-curve intersections'
    else:
        dt = 1/report['fps'] if report.get('fps') is not None else 1.
        for frame in report['frames']:
            frame['time'] = frame.get('frame_offset', frame['frame'])*dt
        report.update(time_unit='seconds' if report.get('fps') is not None else 'frames',
                      velocity_unit='pixels/second' if report.get('fps') is not None else 'pixels/frame',
                      acceleration_unit='pixels/second^2' if report.get('fps') is not None else 'pixels/frame^2')
    return report


def export_report(report, results, output, path=None):
    from calibration import load_calibration
    calibration = load_calibration(output)
    report = curve_crossing_snapshot(report)
    path = Path(path) if path is not None else Path(output)/'nodes.csv'
    paths = [path, path.with_name(path.stem+'_curves.csv'), path.with_name(path.stem+'_candidates.csv')]
    rows = [[], [], []]
    results = {r['frame']: r for r in results}
    scale = calibration['units_per_pixel'] if calibration is not None else None
    unit = calibration['unit'] if calibration is not None else None
    time_unit = 'second' if report.get('fps') is not None else 'frame'
    for frame in report['frames']:
        base = dict(frame=frame['frame'], frame_offset=frame.get('frame_offset', frame['frame']), source=frame['source'],
                    time=frame['time'], time_unit=report['time_unit'], fps=report.get('fps'), human_verified=frame['human_verified'],
                    id_scope='video' if report.get('identities_linked', True) else 'frame',
                    needs_reanalysis=report.get('needs_reanalysis', False), length_unit=unit, units_per_pixel=scale)
        base['detection_issues'] = '; '.join(results[frame['frame']].get('detection_issues', []))
        def physical(row, x, y):
            row.update(x_physical=x*scale if scale is not None and x is not None else None,
                       y_physical=y*scale if scale is not None and y is not None else None)
            return row
        for node in frame['nodes']:
            row = {**base, **{k: node.get(k) for k in ('id', 'row', 'column', 'kind', 'side', 'x', 'y', 'status', 'measurement_source',
                                                      'estimate_source', 'position_verified', 'kinematics_uses_estimates', 'intersection_count')}}
            row['observed'] = node.get('measured_xy') is not None
            physical(row, node.get('x'), node.get('y'))
            for field in ('velocity', 'acceleration'):
                value = node.get(field)
                for axis, index in (('x', 0), ('y', 1)):
                    row[field+'_'+axis] = value[index] if value is not None else None
                    row[field+'_physical_'+axis] = value[index]*scale if value is not None and scale is not None else None
                row[field+'_unit'] = report[field+'_unit']
                row[field+'_physical_unit'] = f'{unit}/{time_unit}'+('^2' if field == 'acceleration' else '') if scale is not None else None
            row['details_json'] = json.dumps(node, allow_nan=False)
            rows[0].append(row)
        curves, ignored = frame['curves'], frame['ignored_curves']
        for curve in curves+ignored:
            for control, xy in enumerate(curve.get('control_points', [[None, None]])):
                row = dict(**base, family=curve['family'], line_id=curve['line_id'], control=control,
                           x=xy[0], y=xy[1], rmse=curve.get('rmse'), details_json=json.dumps(curve, allow_nan=False))
                rows[1].append(physical(row, *xy))
        for candidate in results[frame['frame']]['candidate_nodes']:
            row = dict(**base, detection_id=candidate['id'], x=candidate['x'], y=candidate['y'], kind=candidate['kind'],
                       spatial_accepted=candidate.get('spatial_accepted'), details_json=json.dumps(candidate, allow_nan=False))
            rows[2].append(physical(row, candidate['x'], candidate['y']))
    for target, table, default in zip(paths, rows, ('id', 'line_id', 'details_json')):
        target.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(k for row in table for k in row)) or ['frame', default]
        with target.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    return paths


def export_detections(results, output, nodes, fps):
    """Before first-frame approval, fit and export frame-local curve crossings."""
    from fit_bezier_curves import fit_labeled_curves
    frames = []
    for result in sorted(results, key=lambda r: r['frame']):
        draft = nodes if result['frame'] == 0 else [label_node(n, result) for n in result['final_nodes']]
        frames.append(dict(frame=result['frame'], frame_offset=result.get('frame_offset', result['frame']),
                           source=result['source'], image_size=result['image_size'], human_verified=False,
                           nodes=[dict(n, measured_xy=[n['x'], n['y']], status='draft' if result['frame'] == 0 else 'detected',
                                       measurement_source='human_draft' if result['frame'] == 0 else 'independent_detection') for n in draft],
                           curves=[], ignored_curves=[]))
        frames[-1]['curves'], frames[-1]['ignored_curves'] = fit_labeled_curves(frames[-1]['nodes'], result['image_size'], result['width'])
    return export_report(dict(fps=fps, identities_linked=False, frames=frames), results, output)

