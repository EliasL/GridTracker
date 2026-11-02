"""Resolve duplicate grid identities without treating deformed strings as rigid."""
from collections import defaultdict
import numpy as np

from mesh_hints import identity


def resolve_conflicts(nodes, result, preferred=()):
    groups = defaultdict(list)
    for node in nodes:
        groups[identity(node)].append(node)
    preferred = set(preferred)
    retained, rejected = [], []
    for key, candidates in groups.items():
        if len(candidates) == 1:
            retained += candidates
            continue
        human = [n for n in candidates if n['id'] in preferred or n.get('human_edited')]
        if len(human) > 1:
            latest = [n for n in human if n['id'] in preferred]
            if len(latest) != 1:
                raise ValueError(f'Two human edits have the same grid identity {key}; relabel one of them')
            human = latest
        def score(node):
            residuals = []
            for label, along, across, axis in (('row', 'x', 'y', 'y'), ('column', 'y', 'x', 'x')):
                if node[label] is None:
                    continue
                others = [n for n in nodes if identity(n) != key and n[label] == node[label]]
                if others:
                    ordered = sorted(others, key=lambda n: n[along])
                    expected = np.interp(node[along], [n[along] for n in ordered], [n[across] for n in ordered])
                else:
                    peaks = result['maxima'][axis]['clean']
                    expected = peaks[node[label]] if node[label] < len(peaks) else node[across]
                residuals.append(abs(node[across]-expected)/result['width'])
            return (sum(residuals), -float(node.get('correlation') or node.get('peak') or 0), node['id'])
        winner = human[0] if human else min(candidates, key=score)
        retained.append(dict(winner, resolved_duplicate_ids=[n['id'] for n in candidates if n is not winner]))
        rejected += [dict(n, conflict_reason='duplicate_grid_identity', retained_id=winner['id'])
                     for n in candidates if n is not winner]
    ids = [n['id'] for n in retained]
    if len(ids) != len(set(ids)):
        raise ValueError('One node ID refers to different grid identities')
    return retained, rejected
