"""Advisory missing crossings inferred from the current boundary endpoints."""
import numpy as np


def identity(node):
    return (node['kind'], node.get('side') if node['kind'] == 'edge' else None,
            node['row'], node['column'])


def string_models(nodes, size):
    """Piecewise linear strings through current nodes; boundary labels set extent."""
    models = {}
    for label, across, along in (('column', 'x', 'y'), ('row', 'y', 'x')):
        labels = {n[label] for n in nodes if n['kind'] == 'edge' and n[label] is not None}
        strings = {}
        for key in labels:
            points = sorted((n[along], n[across]) for n in nodes if n[label] == key)
            positions = np.array(sorted({p[0] for p in points}))
            values = np.array([np.mean([v for p, v in points if p == at]) for at in positions])
            if len(positions) == 1:
                positions = np.array([0., size[1 if label == 'column' else 0]-1.])
                values = np.repeat(values, 2)
            strings[key] = (positions, values)
        models[label] = strings
    return models


def missing_crossings(nodes, size, width):
    models = string_models(nodes, size)
    occupied = {(n['row'], n['column']) for n in nodes if n['kind'] == 'interior'}
    # Suppress suggestions near a cropped boundary, even if labels imply a cell.
    margin = 2*width
    hints = []
    for row, horizontal in models['row'].items():
        for column, vertical in models['column'].items():
            if (row, column) in occupied:
                continue
            xy = np.array([np.mean(vertical[1]), np.mean(horizontal[1])])
            for _ in range(12):
                next_xy = np.array([np.interp(xy[1], *vertical), np.interp(xy[0], *horizontal)])
                if np.linalg.norm(next_xy-xy) < .01:
                    xy = next_xy
                    break
                xy = next_xy
            # Only flag a plausible intersection, not a failed fixed-point solve.
            residual = np.linalg.norm(xy-[np.interp(xy[1], *vertical), np.interp(xy[0], *horizontal)])
            if residual <= width and np.all(xy >= margin) and np.all(xy <= np.array(size)-1-margin):
                hints.append(dict(row=row, column=column, x=float(xy[0]), y=float(xy[1])))
    return hints


def label_from_mesh(node, nodes, result):
    """Reuse a nearby string identity, or allocate a new identity for a new edge."""
    from reference_editor import label_node
    node = label_node(node, result)
    models = string_models(nodes, result['image_size'])
    for label, across, along in (('column', 'x', 'y'), ('row', 'y', 'x')):
        if node[label] is None or not models[label]:
            continue
        distances = {key: abs(float(np.interp(node[along], *model))-node[across]) for key, model in models[label].items()}
        nearest = min(distances, key=distances.get)
        positions = sorted(float(np.mean(m[1])) for m in models[label].values())
        pitch = float(np.median(np.diff(positions))) if len(positions) > 1 else 6*result['width']
        if node['kind'] == 'edge' and distances[nearest] > max(2*result['width'], .35*pitch):
            node[label] = max(n[label] for n in nodes if n[label] is not None)+1
        else:
            node[label] = nearest
    return node
