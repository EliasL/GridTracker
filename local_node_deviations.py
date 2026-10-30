"""Group nodes between line-scan minima and calculate within-group coordinate z-scores."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=Path('output/completed_edge_nodes/cleaned/analysis.json'))
    parser.add_argument('--output',type=Path,default=Path('output/local_node_deviations'))
    args=parser.parse_args()
    data=json.loads(args.analysis.read_text());nodes=data['nodes']
    with Image.open(data['source']) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
    results={axis:group_deviations(nodes,axis,extent,data['sigma_pixels']) for axis,extent in zip(('x','y'),image.size)}
    args.output.mkdir(parents=True,exist_ok=True)
    records=[record for result in results.values() for record in result[-1]]
    with (args.output/'node_deviations.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    output=dict(data,local_group_statistics={axis:{'minima':result[2].tolist(),'groups':result[3],
                        'node_deviations':result[4]} for axis,result in results.items()},
                local_deviation_definition='absolute coordinate distance to own group mean / population group SD',
                local_deviation_filter_applied=False,
                local_deviation_boundary_rule='[left,right); image edges close outermost groups; exact-minimum nodes go right')
    (args.output/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'text.usetex':False,'font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(2,2,figsize=(16,10),constrained_layout=True)
    max_z=max(record['distance_in_sd'] or 0 for record in records)
    vmax=max(3,float(np.ceil(max_z)))
    background=ImageEnhance.Brightness(image).enhance(3)
    lookup={node['id']:node for node in nodes}
    for column,(axis,result) in enumerate(results.items()):
        positions,signal,minima,groups,deviations=result
        graph,real=axes[:,column]
        defined=[record for record in deviations if record['distance_in_sd'] is not None]
        undefined=[record for record in deviations if record['distance_in_sd'] is None]
        for group in groups:
            if group['group']%2==0:graph.axvspan(group['left'],group['right'],color='gray',alpha=.08)
        for boundary in minima:graph.axvline(boundary,color='gray',ls=':',lw=.8)
        scatter=graph.scatter([r['coordinate'] for r in defined],[r['distance_in_sd'] for r in defined],
                              c=[r['distance_in_sd'] for r in defined],cmap='viridis',vmin=0,vmax=vmax,s=25)
        if undefined:graph.scatter([r['coordinate'] for r in undefined],np.zeros(len(undefined)),marker='x',color='gray',label='SD undefined')
        for z in (1,2,3):graph.axhline(z,color='gray',ls='--',alpha=.4)
        graph.set(xlabel=f'Node {axis}-coordinate (px)',ylabel='Distance from own group mean (SD)',
                  title=f'{axis}-direction: {len(groups)} groups, {len(deviations)} eligible nodes',xlim=(0,positions[-1]))
        graph.grid(axis='y',alpha=.2)
        real.imshow(background)
        real.scatter([lookup[r['id']]['x'] for r in defined],[lookup[r['id']]['y'] for r in defined],
                     c=[r['distance_in_sd'] for r in defined],cmap='viridis',vmin=0,vmax=vmax,
                     marker='o',s=34,edgecolors='black',linewidths=.4,clip_on=False)
        if undefined:real.scatter([lookup[r['id']]['x'] for r in undefined],[lookup[r['id']]['y'] for r in undefined],marker='x',color='gray')
        real.set_title(f'Nodes colored by {axis}-group deviation');real.set_axis_off()
    fig.colorbar(scatter,ax=axes.ravel().tolist(),shrink=.8,label='Absolute within-group z-score')
    fig.suptitle('Group boundaries = all local scan minima; means and SDs computed from node coordinates within each group; no removals')
    fig.savefig(args.output/'local_deviations.png',dpi=180)
    fig.savefig(args.output/'local_deviations.pdf');plt.close(fig)
    print(json.dumps({axis:{'groups':len(result[3]),'nodes':len(result[4]),
        'group_sizes':[g['count'] for g in result[3]],
        'undefined_sd':sum(r['signed_z'] is None for r in result[4]),
        'above_2sd':sum((r['distance_in_sd'] or 0)>2 for r in result[4]),
        'above_3sd':sum((r['distance_in_sd'] or 0)>3 for r in result[4])}
        for axis,result in results.items()},indent=2))


if __name__=='__main__':
    main()
