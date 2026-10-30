"""Node identities, validation, hashes, and atomic JSON persistence."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

GEOMETRY_FIELDS = ('x', 'y', 'row', 'column', 'kind', 'side', 'scan_axes')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analysis_hash(result):
    fields = {k: result[k] for k in ('final_nodes', 'maxima', 'width')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def label_node(node, result):
    node = deepcopy(node)
    edge = node.get('kind') == 'edge'
    for axis, name in (('x', 'column'), ('y', 'row')):
        eligible = not edge or axis in node['scan_axes']
        peaks = np.asarray(result['maxima'][axis]['clean'])
        node[name] = int(np.argmin(abs(peaks-node[axis]))) if eligible else None
        if eligible and len(peaks) >= 2:
            pitch = float(np.median(np.diff(peaks)))
            end = result['image_size'][0 if axis == 'x' else 1]-1
            border = max(2*result['width'], .3*pitch)
            if node[axis] > peaks[-1]+pitch/2 and end-node[axis] < border:
                node[name] = len(peaks)  # A partially visible extra string.
            elif node[axis] < peaks[0]-pitch/2 and node[axis] < border:
                node[name] = len(peaks)+1
    return node


def validate_nodes(nodes, size, require_crossing=True):
    if require_crossing and (not nodes or not any(n['kind'] == 'interior' for n in nodes)):
        raise ValueError('The reference must contain at least one crossing')
    seen, ids = set(), set()
    for n in nodes:
        if n['id'] in ids:
            raise ValueError(f"Duplicate node ID {n['id']}")
        ids.add(n['id'])
        if not np.isfinite([n['x'], n['y']]).all() or not (0 <= n['x'] < size[0] and 0 <= n['y'] < size[1]):
            raise ValueError(f"Node {n['id']} is outside the image")
        if n['kind'] == 'interior':
            key = ('crossing', n['row'], n['column'])
            values = (n['row'], n['column'])
        elif n['kind'] == 'edge':
            side = n['side']
            if side not in ('top', 'bottom', 'left', 'right'):
                raise ValueError(f'Unknown edge {side}')
            axis = 'x' if side in ('top', 'bottom') else 'y'
            label = 'column' if axis == 'x' else 'row'
            normal = 'y' if axis == 'x' else 'x'
            boundary = 0 if side in ('top', 'left') else size[1 if normal == 'y' else 0]-1
            if n[normal] != boundary or n['scan_axes'] != [axis] or n['row' if label == 'column' else 'column'] is not None:
                raise ValueError(f"Edge node {n['id']} has inconsistent boundary metadata")
            key, values = (side, n[label]), (n[label],)
        else:
            raise ValueError(f"Unknown node kind {n['kind']}")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
            raise ValueError(f"Node {n['id']} needs nonnegative integer row/column labels")
        if key in seen:
            raise ValueError(f'Duplicate grid identity {key}; move, relabel, or delete a node')
        seen.add(key)
    interior = [n for n in nodes if n['kind'] == 'interior']
    for label, order, coordinate in (('row', 'column', 'x'), ('column', 'row', 'y')):
        directions = {}
        for group in {n[label] for n in interior}:
            line = sorted((n for n in interior if n[label] == group), key=lambda n: n[order])
            for i, a in enumerate(line):
                for b in line[i+1:]:
                    pair = (a[order], b[order])
                    direction = np.sign(b[coordinate]-a[coordinate])
                    if direction == 0 or (pair in directions and directions[pair] != direction):
                        raise ValueError(f'Crossed identities in {label} {group}; correct row/column labels')
                    directions[pair] = direction


def parse_fps(value):
    if value is None or str(value).strip() == '':
        return None
    fps = float(value)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Frame sampling rate must be positive and finite, or blank')
    return fps


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_reference(path, result, nodes, approved, fps=None, *, editor_state=None):
    if approved:
        validate_nodes(nodes, result['image_size'], require_crossing=result.get('frame', 0) == 0)
        nodes = deepcopy(nodes)
        for node in nodes:
            if 'measured_xy' in node:
                node.update(measured_xy=[node['x'], node['y']], velocity=None, acceleration=None,
                            status='verified', measurement_source='human')
    document = dict(source=result['source'], source_sha256=file_hash(result['source']),
                    analysis_sha256=analysis_hash(result), image_size=result['image_size'],
                    approved=approved, approved_at=datetime.now(timezone.utc).isoformat() if approved else None,
                    nodes=nodes, fps=parse_fps(fps), frame=result.get('frame', 0))
    if editor_state is not None:
        document['editor_state'] = editor_state
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, document)
    return document


def validate_reference(document, result):
    if document.get('approved') is not True or not document.get('approved_at'):
        raise ValueError('The first frame has not been explicitly approved')
    if document['source_sha256'] != file_hash(result['source']) or document['analysis_sha256'] != analysis_hash(result):
        raise ValueError('The approved reference belongs to different image pixels or detection results')
    if document['image_size'] != result['image_size']:
        raise ValueError('Reference image dimensions changed')
    validate_nodes(document['nodes'], result['image_size'], require_crossing=result.get('frame', 0) == 0)
    return document


def load_reference(path, result):
    return validate_reference(json.loads(Path(path).read_text()), result)


def save_ui_state(output, frame, mode):
    write_json(Path(output)/'ui_state.json', dict(frame=frame, mode=mode))


def load_ui_state(output, frame_count):
    path = Path(output)/'ui_state.json'
    state = dict(frame=0, mode='review')
    if path.exists():
        state = json.loads(path.read_text())
    else:
        # Older versions saved drafts but no cursor. Recover the newest
        # unfinished edit, or the most recently confirmed frame.
        fields = ('id',)+GEOMETRY_FIELDS
        def signature(document):
            return [{k: n.get(k) for k in fields} for n in document['nodes']], document.get('fps')
        for draft_path in sorted((Path(output)/'reviews').glob('frame_*.draft.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            draft = json.loads(draft_path.read_text())
            approved_path = draft_path.with_name(draft_path.name.replace('.draft.json', '.json'))
            if not approved_path.exists() or signature(draft) != signature(json.loads(approved_path.read_text())):
                state = dict(frame=draft['frame'], mode='edit')
                break
        else:
            review = Path(output)/'review.json'
            if review.exists():
                anchors = json.loads(review.read_text())['anchors']
                latest = max(anchors, key=lambda i: anchors[i]['approved_at'])
                state['frame'] = int(latest)
    if type(state['frame']) is not int or not 0 <= state['frame'] < frame_count or state['mode'] not in ('review', 'edit'):
        raise ValueError('Saved interface state does not match this frame sequence')
    return state


def node_geometry(nodes):
    return {n['id']: {k: n.get(k) for k in GEOMETRY_FIELDS} for n in nodes}


def load_draft(path, result):
    document = json.loads(Path(path).read_text())
    if document['source_sha256'] != file_hash(result['source']) or document['analysis_sha256'] != analysis_hash(result):
        raise ValueError('Saved editing draft belongs to different image pixels or detection results')
    return document

