"""Group nodes between line-scan minima and calculate within-group coordinate z-scores."""

import numpy as np
from scipy.signal import find_peaks

from scan_node_lines import line_scan


def group_deviations(nodes, axis, extent, sigma):
    eligible=[node for node in nodes if axis in node.get('scan_axes',['x','y'])]
    coordinates=np.array([node[axis] for node in eligible])
    positions,signal=line_scan(coordinates,extent,sigma)
    minima,_=find_peaks(-signal)
    boundaries=np.r_[0,minima,extent]
    assigned=np.searchsorted(minima,coordinates,side='right')
    records,groups=[],[]
    for group in range(len(boundaries)-1):
        indices=np.flatnonzero(assigned==group)
        if not len(indices):
            continue
        values=coordinates[indices]
        mean,std=float(values.mean()),float(values.std(ddof=0))
        groups.append({'group':group,'left':int(boundaries[group]),'right':int(boundaries[group+1]),
                       'count':len(indices),'mean':mean,'population_std':std})
        for index in indices:
            value=float(coordinates[index])
            z=(value-mean)/std if std>0 else None
            records.append({'id':eligible[index]['id'],'axis':axis,'coordinate':value,'group':group,
                            'group_count':len(indices),'mean':mean,'population_std':std,
                            'signed_z':z,'distance_in_sd':abs(z) if z is not None else None,
                            'status':'defined' if z is not None else 'zero_group_variance'})
    return positions,signal,minima,groups,records


def local_outlier_ids(deviations, cutoff=2):
    """One pass: reject at or above cutoff in any eligible axis; undefined SD stays."""
    if not np.isfinite(cutoff) or cutoff<=0:
        raise ValueError('Cutoff must be finite and positive')
    return {record['id'] for record in deviations if record['distance_in_sd'] is not None
            and record['distance_in_sd']>=cutoff}
