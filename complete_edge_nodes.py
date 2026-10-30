"""Retain border proposals and complete missing string endpoints."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


def complete_endpoints(lines, accepted, profile, cross_profile, allow_missing=False):
    """One endpoint per known line; accepted proposals are retained without moving."""
    lines=np.asarray(lines,dtype=float)
    centers=np.array([node['profile_center'] for node in accepted])
    pitch=float(np.median(np.diff(lines)))
    assignments={}
    if len(centers):
        rows,cols=linear_sum_assignment(abs(lines[:,None]-centers[None,:]))
        for row,col in zip(rows,cols):
            if abs(lines[row]-centers[col])<=pitch/2:
                assignments[int(row)]=int(col)
    records=[dict(node,line_index=None,origin='accepted') for node in accepted]
    for line,index in assignments.items():
        records[index]['line_index']=line
    candidates=[]
    for origin,score in (('edge_score',profile),('cross_score',cross_profile)):
        peaks,_=find_peaks(score)
        candidates.extend((int(p),float(score[p]),origin) for p in peaks if score[p]>0)
    for index,target in enumerate(lines):
        if index in assignments:
            continue
        left=(lines[index-1]+target)/2 if index else max(0,target-pitch/2)
        right=(lines[index+1]+target)/2 if index+1<len(lines) else min(len(profile)-1,target+pitch/2)
        options=[candidate for candidate in candidates if left<=candidate[0]<right]
        if not options:
            if allow_missing:
                records.append({'profile_center':None,'line_index':index,'origin':'unsupported',
                                'straight_line_position':float(target),'reason':'no local maximum inside line search interval'})
                continue
            raise ValueError(f'No local edge/cross maximum near line {index} at {target:g}')
        position,score,origin=min(options,key=lambda item:(abs(item[0]-target),-item[1],item[2]))
        records.append({'profile_center':float(position),'line_index':index,'origin':origin,
                        'peak_score':score,'straight_line_position':float(target),
                        'distance_from_line':float(abs(position-target))})
    if sorted(node['line_index'] for node in records if node['line_index'] is not None)!=list(range(len(lines))):
        raise ValueError('Line endpoints are not uniquely assigned')
    return records
