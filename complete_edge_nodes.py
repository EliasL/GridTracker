"""Accept border proposals, complete missing line endpoints, and export pre-cleanup nodes."""
import argparse
import json
from pathlib import Path

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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edges',type=Path,default=Path('output/edge_nodes'))
    parser.add_argument('--interior',type=Path,default=Path('output/crossing_nodes_lower_mass/nodes.json'))
    parser.add_argument('--cross-scores',type=Path,default=Path('output/color_grayscale_comparison/crossing_scores.npy'))
    parser.add_argument('--output',type=Path,default=Path('output/completed_edge_nodes'))
    args=parser.parse_args()
    edge_data=json.loads((args.edges/'nodes.json').read_text())
    interior=json.loads(args.interior.read_text())
    if edge_data['source']!=interior['source']:
        raise ValueError('Interior and edge nodes refer to different images')
    cross=np.load(args.cross_scores)
    height,image_width=cross.shape
    depth=edge_data['inward_depth']
    # Cross fallback is the strongest score in the same narrow inward border strip.
    cross_profiles={'top':cross[:depth].max(axis=0),'bottom':cross[-depth:].max(axis=0),
                    'left':cross[:,:depth].max(axis=1),'right':cross[:,-depth:].max(axis=1)}
    edge_nodes=[]
    for side in ('top','bottom','left','right'):
        profile=np.loadtxt(args.edges/f'{side}_scores.csv',delimiter=',',skiprows=1)[:,1]
        lines=edge_data['grid_vertical_x'] if side in ('top','bottom') else edge_data['grid_horizontal_y']
        accepted=[node for node in edge_data['edge_nodes'] if node['side']==side]
        records=complete_endpoints(lines,accepted,profile,cross_profiles[side])
        for record in records:
            position=record['profile_center']
            x,y=((position,0) if side=='top' else (position,height-1) if side=='bottom' else
                 (0,position) if side=='left' else (image_width-1,position))
            edge_nodes.append(dict(record,side=side,x=float(x),y=float(y),kind='edge',
                                   scan_axes=['x'] if side in ('top','bottom') else ['y']))
    original=[dict(node,kind='interior',scan_axes=['x','y']) for node in interior['nodes']]
    next_id=max(node['id'] for node in original)+1
    for offset,node in enumerate(edge_nodes):
        node['id']=next_id+offset
    output={'source':interior['source'],'nodes':original+edge_nodes,'interior_count':len(original),
            'edge_count':len(edge_nodes),'accepted_edge_count':sum(n['origin']=='accepted' for n in edge_nodes),
            'added_edge_count':sum(n['origin']!='accepted' for n in edge_nodes),
            'cross_fallback':f'max over existing {depth}-pixel inward strip, projected onto physical border',
            'matching':'nearest one-to-one accepted match within half median spacing; missing endpoints use nearest local maximum in their line cell',
            'grid_vertical_x':edge_data['grid_vertical_x'],'grid_horizontal_y':edge_data['grid_horizontal_y']}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'nodes.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({key:output[key] for key in ('interior_count','edge_count','accepted_edge_count','added_edge_count')},indent=2))


if __name__=='__main__':
    main()
