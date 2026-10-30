"""Remove nodes unsupported by another node in a scan-coordinate neighborhood."""
import argparse
import json
from pathlib import Path

import numpy as np


def remove_isolated(nodes, axis, radius):
    if axis not in ('x','y') or not np.isfinite(radius) or radius<=0:
        raise ValueError('Require axis x/y and a finite positive neighborhood radius')
    eligible=[node for node in nodes if axis in node.get('scan_axes',['x','y'])]
    if not eligible:
        raise ValueError('No nodes eligible for this scan direction')
    positions=np.array([node[axis] for node in eligible],dtype=float)
    distances=abs(positions[:,None]-positions[None,:])
    np.fill_diagonal(distances,np.inf)
    counts=(distances<=radius).sum(axis=1)
    rejected_ids={node['id'] for node,count in zip(eligible,counts) if count==0}
    kept=[node for node in nodes if node['id'] not in rejected_ids]
    rejected=[node for node in nodes if node['id'] in rejected_ids]
    support=[{'id':node['id'],'neighbors':int(count),'nearest_distance':float(row.min()) if np.isfinite(row.min()) else None}
             for node,count,row in zip(eligible,counts,distances)]
    return kept,rejected,support


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=Path('output/completed_edge_nodes/cleaned/analysis.json'))
    parser.add_argument('--output',type=Path,default=Path('output/group_supported_nodes'))
    parser.add_argument('--axis',choices=['x','y'],default='y')
    parser.add_argument('--radius',type=float,help='Default: twice the scan Gaussian sigma')
    args=parser.parse_args()
    data=json.loads(args.analysis.read_text())
    radius=2*data['sigma_pixels'] if args.radius is None else args.radius
    kept,rejected,support=remove_isolated(data['nodes'],args.axis,radius)
    output=dict(data,nodes=kept,isolation_removed_nodes=rejected,
                isolation_filter={'axis':args.axis,'radius_pixels':radius,'minimum_other_neighbors':1,
                                  'support':support,'source_analysis':str(args.analysis.resolve())})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'retained':len(kept),'removed':rejected,'axis':args.axis,'radius':radius},indent=2))


if __name__=='__main__':
    main()
