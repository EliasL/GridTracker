"""Detect border string endpoints and display the cleaned-maxima straight grid."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import correlate1d

from crossing_nodes import choose_threshold, component_nodes


def edge_line_score(profile, width):
    """Gaussian string-width center minus flanking background, with no invented pixels."""
    if width <= 0 or not np.isfinite(width):
        raise ValueError("String width must be finite and positive")
    radius = max(6, int(np.ceil(3*width)))
    offsets = np.arange(-radius, radius+1)
    center = np.exp(-0.5*(offsets/(width/2.35482))**2)
    flanks = (abs(offsets) >= width).astype(float)
    responses = []
    for kernel in (center, flanks):
        kernel = kernel/kernel.sum()
        support = correlate1d(np.ones_like(profile), kernel, mode="constant", cval=0)
        responses.append(correlate1d(profile,kernel,mode="constant",cval=0)/support)
    positive, background = responses
    return np.clip((positive-background)/np.maximum(1-background,1e-8),0,1)


def histogram_threshold(values):
    """Otsu between-class variance, including zero edge scores, for proposal selection."""
    counts,edges=np.histogram(values,bins=256,range=(0,1))
    centers=(edges[:-1]+edges[1:])/2
    weight=np.cumsum(counts);moment=np.cumsum(counts*centers)
    other=weight[-1]-weight
    valid=(weight>0)&(other>0)
    if not valid.any():raise ValueError('No separable edge-score histogram')
    criterion=np.full(256,-np.inf)
    criterion[valid]=(moment[-1]*weight[valid]-moment[valid]*weight[-1])**2/(weight[valid]*other[valid])
    return float(centers[np.argmax(criterion)])


def detect_edges(field, width, threshold=None, threshold_method="slope"):
    if field.ndim != 2 or not np.isfinite(field).all() or np.any((field<0)|(field>1)):
        raise ValueError("Expected a finite 2D scalar field in [0,1]")
    height, image_width = field.shape
    depth = min(int(np.ceil(3*width)), height, image_width)
    if depth < 1:
        raise ValueError("Invalid edge support depth")
    # Inward weights use only real pixels; stronger near-border support gets more weight.
    weights = np.exp(-np.arange(depth)/width)
    weights /= weights.sum()
    profiles = {"top":weights@field[:depth], "bottom":weights@field[-depth:][::-1],
                "left":field[:,:depth]@weights, "right":field[:,-depth:][:,::-1]@weights}
    scores = {side:edge_line_score(profile,width) for side,profile in profiles.items()}
    if threshold_method not in ('slope','histogram'):
        raise ValueError('Unknown edge threshold method')
    pooled=np.concatenate(list(scores.values()))
    selection=choose_threshold(pooled[None,:]) if threshold is None and threshold_method=='slope' else None
    if threshold is None:threshold=selection['threshold'] if selection else histogram_threshold(pooled)
    nodes = []
    for side, score in scores.items():
        _, components = component_nodes(score[None,:],threshold)
        for component in components:
            position = component['x']
            x, y = ((position,0) if side=='top' else (position,height-1) if side=='bottom' else
                    (0,position) if side=='left' else (image_width-1,position))
            nodes.append({'edge_id':f"{side}-{component['id']}",'side':side,'x':float(x),'y':float(y),'profile_center':position,
                          'support_length':component['pixels'],'peak_score':component['peak'],
                          'score_mass':component['score_mass']})
    return scores, nodes, float(threshold), depth, selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scalar',type=Path,default=Path('output/color_grayscale_comparison/grayscale.npy'))
    parser.add_argument('--model',type=Path,default=Path('output/color_grayscale_comparison/model.json'))
    parser.add_argument('--analysis',type=Path,default=Path('output/cleaned_scan_peaks/analysis.json'))
    parser.add_argument('--threshold',type=float,help='Optional override; default applies the existing slope criterion to edge scores')
    parser.add_argument('--output',type=Path,default=Path('output/edge_nodes'))
    args=parser.parse_args()
    field=np.load(args.scalar)
    model=json.loads(args.model.read_text());data=json.loads(args.analysis.read_text())
    if Path(model['source']).resolve()!=Path(data['source']).resolve():
        raise ValueError('Scalar model and nodes refer to different source images')
    with Image.open(model['source']) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
    if image.size!=field.shape[::-1]:
        raise ValueError('Image and scalar dimensions differ')
    width=model['width_fwhm']
    scores,nodes,threshold,depth,selection=detect_edges(field,width,args.threshold)
    args.output.mkdir(parents=True,exist_ok=True)
    for side,score in scores.items():
        np.savetxt(args.output/f'{side}_scores.csv',np.column_stack((np.arange(len(score)),score)),
                   delimiter=',',header='position,edge_score',comments='')
    # Display the 1D edge scores in thin border bands, not as interior detections.
    heat=np.zeros_like(field)
    band=5
    heat[:band]=scores['top'][None,:];heat[-band:]=scores['bottom'][None,:]
    heat[:,:band]=np.maximum(heat[:,:band],scores['left'][:,None])
    heat[:,-band:]=np.maximum(heat[:,-band:],scores['right'][:,None])
    np.save(args.output/'edge_heatmap.npy',heat)
    x_lines=data['scans']['x']['retained_peaks'];y_lines=data['scans']['y']['retained_peaks']
    output={'source':model['source'],'width_fwhm':width,'inward_depth':depth,'threshold':threshold,
            'threshold_selection':selection,'display_border_band':band,'edge_nodes':nodes,
            'existing_nodes':data['nodes'],'grid_vertical_x':x_lines,'grid_horizontal_y':y_lines,
            'filtering':'none; proposals only, no mass cutoff or final unwanted-node filtering',
            'center':'geometric centroid of each connected above-threshold 1D edge region'}
    (args.output/'nodes.json').write_text(json.dumps(output,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    plt.rcParams.update({'text.usetex':False,'font.family':'DejaVu Sans'})
    background=ImageEnhance.Brightness(image).enhance(3)
    existing=np.array([[node['x'],node['y']] for node in data['nodes']])
    proposed=np.array([[node['x'],node['y']] for node in nodes]).reshape(-1,2)
    for name,title,grid,draw_nodes in (
        ('edge_heatmap',f'Edge string score; width {width:.2f} px; inward support {depth} px',False,False),
        ('edge_nodes',f'Proposed edge nodes: {len(nodes)}; score threshold {threshold:.3f}',False,True),
        ('straight_grid',f'Straight grid: {len(x_lines)} vertical and {len(y_lines)} horizontal lines',True,True)):
        fig,axis=plt.subplots(figsize=(13,7),constrained_layout=True)
        axis.imshow(background)
        if not grid:
            selected=heat>threshold if draw_nodes else heat>0
            overlay=axis.imshow(heat,cmap='turbo',vmin=0,vmax=1,alpha=.85*selected)
            fig.colorbar(plt.cm.ScalarMappable(norm=overlay.norm,cmap=overlay.cmap),ax=axis,
                         shrink=.75,label='Edge score (1D response drawn in 5 px border bands)')
        else:
            for x in x_lines: axis.axvline(x,color='cyan',lw=.8,alpha=.8)
            for y in y_lines: axis.axhline(y,color='magenta',lw=.8,alpha=.8)
        if draw_nodes:
            axis.scatter(existing[:,0],existing[:,1],marker='+',s=28,color='white',label=f'Existing nodes: {len(existing)}')
            marker=axis.scatter(proposed[:,0],proposed[:,1],marker='o',s=65,facecolors='none',
                                edgecolors='yellow',linewidths=1.5,clip_on=False,label=f'Proposed edge nodes: {len(nodes)}')
            marker.set_path_effects([patheffects.withStroke(linewidth=3,foreground='black')])
            axis.legend(loc='upper center',bbox_to_anchor=(.5,-.015),ncol=2,facecolor='#222222',labelcolor='white')
        axis.set_title(title);axis.set_axis_off()
        fig.savefig(args.output/f'{name}.png',dpi=180);fig.savefig(args.output/f'{name}.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    for axis,(side,score) in zip(axes.ravel(),scores.items()):
        axis.plot(score)
        axis.axhline(threshold,color='gray',ls='--',label=f'Threshold {threshold:.3f}')
        for node in nodes:
            if node['side']==side: axis.axvline(node['profile_center'],color='orange',alpha=.5)
        axis.set(title=side,xlabel='Position along image edge (px)',ylabel='String score')
        axis.grid(alpha=.2);axis.legend()
    fig.savefig(args.output/'edge_profiles.png',dpi=180);plt.close(fig)
    print(json.dumps({'threshold':threshold,'width':width,'inward_depth':depth,
        'edge_node_counts':{side:sum(n['side']==side for n in nodes) for side in scores},
        'grid_lines':[len(x_lines),len(y_lines)]},indent=2))


if __name__=='__main__':
    main()
