"""Inspect leave-one-out line support after maxima filtering; do not remove nodes."""
import argparse
import json
from pathlib import Path

import numpy as np


def node_line_scores(nodes, sigma):
    if not nodes or not np.isfinite(sigma) or sigma<=0:
        raise ValueError('Require nodes and a finite positive line-score tolerance')
    points=np.array([[node['x'],node['y']] for node in nodes],dtype=float)
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite node coordinates')
    eligible=np.array([[axis in node.get('scan_axes',['x','y']) for axis in ('x','y')] for node in nodes])
    if np.any(eligible.sum(axis=1)==0):
        raise ValueError('Every node must belong to at least one scan direction')
    scores=np.full((len(nodes),2),np.nan)
    for axis in (0,1):
        indices=np.flatnonzero(eligible[:,axis])
        coordinates=points[indices,axis]
        kernel=np.exp(-0.5*((coordinates[:,None]-coordinates[None,:])/sigma)**2)
        np.fill_diagonal(kernel,0)  # A node must not support itself.
        scores[indices,axis]=kernel.sum(axis=1)
    average=np.nansum(scores,axis=1)/eligible.sum(axis=1)
    return scores,average


def summarize(values, nsigma=1):
    values=np.asarray(values)
    values=values[np.isfinite(values)]
    if not len(values):
        raise ValueError('Cannot summarize an empty direction score distribution')
    mean,std=float(values.mean()),float(values.std(ddof=0))
    return {'count':len(values),'mean':mean,'population_std':std,
            'lower':mean-nsigma*std,'upper':mean+nsigma*std,'nsigma':nsigma}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=Path('output/completed_edge_nodes/cleaned/analysis.json'))
    parser.add_argument('--output',type=Path,default=Path('output/node_line_support'))
    args=parser.parse_args()
    data=json.loads(args.analysis.read_text())
    nodes=data['nodes']
    scores,average=node_line_scores(nodes,data['sigma_pixels'])
    statistics={name:summarize(values) for name,values in
                (('x',scores[:,0]),('y',scores[:,1]),('average',average))}
    cutoff=statistics['average']['lower']
    records=[]
    for node,(x,y),value in zip(nodes,scores,average):
        records.append(dict(node,line_score_x=float(x) if np.isfinite(x) else None,
                            line_score_y=float(y) if np.isfinite(y) else None,
                            average_line_score=float(value),below_mean_minus_sd=bool(value<cutoff)))
    output=dict(data,nodes=records,line_score_test={'source_analysis':str(args.analysis.resolve()),
        'stage':'after maxima cleanup; isolated-node filter not applied',
        'definition':'mean of eligible directional leave-one-out Gaussian line support at each node coordinate',
        'statistics':statistics,'flagged_ids':[n['id'] for n in records if n['below_mean_minus_sd']],
        'removal_applied':False})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    import csv
    with (args.output/'scores.csv').open('w') as stream:
        fields=['id','kind','x','y','line_score_x','line_score_y','average_line_score','below_mean_minus_sd']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'text.usetex':False,'font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for axis,name,values,title in zip(axes,('x','y','average'),(scores[:,0],scores[:,1],average),
        ('Vertical-line support at node x','Horizontal-line support at node y','Average eligible line support per node')):
        values=values[np.isfinite(values)];stats=statistics[name]
        axis.hist(values,bins=20,color='steelblue',edgecolor='white')
        axis.axvline(stats['mean'],color='black',label=f"Mean {stats['mean']:.2f}")
        axis.axvline(stats['lower'],color='red',ls='--',label=f"Mean - SD: {stats['lower']:.2f}")
        axis.axvline(stats['upper'],color='gray',ls='--',label=f"Mean + SD: {stats['upper']:.2f}")
        axis.set(title=f"{title}\nN={stats['count']}; population SD={stats['population_std']:.2f}",
                 xlabel='Effective number of OTHER nearby nodes',ylabel='Number of nodes')
        axis.yaxis.set_major_locator(MaxNLocator(integer=True));axis.grid(axis='y',alpha=.2);axis.legend(fontsize=8)
    fig.suptitle(f"Line-score distributions AFTER maxima filtering: {len(nodes)} nodes; no isolation removal; 20 bins")
    fig.savefig(args.output/'line_score_distribution.png',dpi=180)
    fig.savefig(args.output/'line_score_distribution.pdf');plt.close(fig)
    print(json.dumps({'node_count':len(nodes),'statistics':statistics,
        'flagged_count':sum(n['below_mean_minus_sd'] for n in records)},indent=2))


if __name__=='__main__':
    main()
