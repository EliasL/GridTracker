"""Reproducible per-frame pipeline. All numeric libraries run single-threaded."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path(__file__).parent/'.matplotlib-cache')
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance,ImageOps
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex':False,'font.family':'DejaVu Sans','figure.max_open_warning':0})
from gridtracker import ridge_maps,find_seed,estimate_string_width,crossing_evidence,plot_width_diagnostics
from compare_crossing_inputs import profile_color_grayscale
from crossing_nodes import choose_threshold,component_nodes,filter_by_mass
from scan_node_lines import line_scan
from local_peak_spacing import prune_split_peaks,sliding_spacing
from edge_nodes import detect_edges
from complete_edge_nodes import complete_endpoints
from local_node_deviations import group_deviations,local_outlier_ids
from fit_bezier_curves import fit_curve,bezier_basis


def scan_cleanup(nodes,shape,sigma):
    points=np.array([[n['x'],n['y']] for n in nodes])
    xmask=np.array(['x' in n.get('scan_axes',['x','y']) for n in nodes])
    ymask=np.array(['y' in n.get('scan_axes',['x','y']) for n in nodes])
    x,sx=line_scan(points[xmask,0],int(shape[0]),sigma)
    rawx,_=find_peaks(sx);px,_,rx=prune_split_peaks(rawx)
    closest=rawx[np.argmin(abs(points[:,0,None]-rawx),axis=1)]
    keep=~xmask|~np.isin(closest,[r['peak'] for r in rx])
    y,sy=line_scan(points[keep & ymask,1],int(shape[1]),sigma)
    rawy,_=find_peaks(sy);py,_,ry=prune_split_peaks(rawy)
    return [n for n,k in zip(nodes,keep) if k], {'x':(x,sx,rawx,px,rx),'y':(y,sy,rawy,py,ry)}


def run_frame(task, *, fit_curves=True, diagnostics=True):
    index,source,root=task
    out=Path(root)/f'frame_{index:02d}';out.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as original: image=ImageOps.exif_transpose(original).convert('RGB')
    background=ImageEnhance.Brightness(image).enhance(3)
    artifacts=[]
    def save(fig,name,caption):
        path=out/f'{len(artifacts):02d}_{name}.png'
        if diagnostics:
            fig.savefig(path,dpi=135)
            artifacts.append({'path':str(path.resolve()),'title':name.replace('_',' ').title(),'caption':caption})
        plt.close(fig)
    def nodes_plot(nodes,title,grid=None,rejected=None):
        fig,ax=plt.subplots(figsize=(12,6),constrained_layout=True);ax.imshow(background)
        if grid:
            for x in grid['x']:ax.axvline(x,color='cyan',lw=.6)
            for y in grid['y']:ax.axhline(y,color='magenta',lw=.6)
        for edge,color,marker in ((False,'white','+'),(True,'yellow','o')):
            points=np.array([[n['x'],n['y']] for n in nodes if (n.get('kind')=='edge')==edge]).reshape(-1,2)
            ax.scatter(points[:,0],points[:,1],s=28,marker=marker,color=color,clip_on=False)
        if rejected:
            p=np.array([[n['x'],n['y']] for n in rejected]);ax.scatter(p[:,0],p[:,1],s=75,marker='x',color='red')
        ax.set_title(title);ax.set_axis_off();return fig
    def scans_plot(scans):
        fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
        for col,(name,(pos,signal,raw,clean,removed)) in enumerate(scans.items()):
            ax=axes[0,col];ax.plot(pos,signal)
            ax.plot(clean,signal[clean],'o',color='green',label=f'{len(clean)} retained')
            bad=np.array([r['peak'] for r in removed],dtype=int)
            ax.plot(bad,signal[bad],'x',color='red',label=f'{len(bad)} removed');ax.legend()
            ax.set(title=f'{name}-scan',ylabel='Line score')
            ax=axes[1,col]
            for peaks,style,label in ((raw,'--','Before'),(clean,'-','After')):
                _,window,_,spacing,_=sliding_spacing(signal,peaks=peaks,allow_missing=True)
                ax.plot(pos,spacing,style,label=label)
            ax.set(xlabel=f'{name} (px)',ylabel='Mean local spacing (px)',title=f'Quarter-image window: {window:g} px');ax.legend()
        return fig
    maps=ridge_maps(image);seed,selected,candidates=find_seed(maps)
    fig,ax=plt.subplots(figsize=(12,6),constrained_layout=True);ax.imshow(background)
    for lines in (seed,seed.transpose(1,0,2)):
        for line in lines:ax.plot(line[:,0],line[:,1],color='cyan')
    ax.set_axis_off();ax.set_title(f"Selected {selected['corner']} {selected['n']}x{selected['n']} seed")
    save(fig,'seed','Color-neutral contrast establishes a corner seed, expanded until uncertainty exceeds 0.45.')
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for corner in ('top-left','top-right','bottom-left','bottom-right'):
        rows=[r for r in candidates if r['corner']==corner]
        for ax,key in zip(axes,('total','density')):ax.plot([r['n'] for r in rows],[r[key] for r in rows],'-o',label=corner);ax.set(xlabel='n x n crossings',ylabel=key)
    axes[1].axhline(.45,color='black',ls='--');axes[0].legend()
    save(fig,'seed_uncertainty','Total area-integrated uncertainty and uncertainty per unit area, for all four corners.')
    width,measurements=estimate_string_width(image,seed)
    if diagnostics:
        plot_width_diagnostics(image,{'width':width,'width_samples':measurements},out)
        artifacts.append({'path':str((out/'width_diagnostics.png').resolve()),'title':'String width distribution','caption':f'{len(measurements)} accepted profiles; pooled Gaussian FWHM {width:.3f} px. Fits use original pixels.'})
    scalar,color_model=profile_color_grayscale(image,measurements)
    cross=crossing_evidence(scalar,width)
    if diagnostics:
        np.save(out/'scalar.npy',scalar);np.save(out/'crossing_scores.npy',cross)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    axes[0].imshow(scalar,cmap='gray',vmin=0,vmax=1);axes[0].set_title('Learned string-color scalar field')
    axes[1].imshow(background);heat=axes[1].imshow(cross,cmap='turbo',vmin=0,vmax=1,alpha=.85*cross)
    axes[1].set_title('Four-arm crossing score overlay')
    for ax in axes:ax.set_axis_off()
    fig.colorbar(plt.cm.ScalarMappable(norm=heat.norm,cmap=heat.cmap),ax=axes[1],shrink=.65)
    save(fig,'scalar_and_crossings',f'Same measured-width filter: {2*int(np.ceil(3*width))+1} px window; four arms and off-axis negative support.')
    selection=choose_threshold(cross,allow_unsustained=True);threshold=selection['threshold'];mask,all_nodes=component_nodes(cross,threshold)
    interior,small,mass=filter_by_mass(all_nodes,mass_key='pixels',upper_limit=False)
    for n in interior:n.update(kind='interior',scan_axes=['x','y'])
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    x=np.array(selection['thresholds']);axes[0].plot(x,10**np.array(selection['smoothed_log_fraction']));axes[0].set_yscale('log')
    axes[0].axvline(threshold,color='red');axes[0].set(xlabel='Crossing threshold',ylabel='Fraction of entire image retained')
    areas=[n['pixels'] for n in all_nodes];axes[1].hist(areas,bins=20);axes[1].axvline(mass['lower'],color='red');axes[1].set(xlabel='Region pixel mass',ylabel='Count')
    save(fig,'threshold_and_mass',f'Threshold {threshold:.5f}; {len(all_nodes)} components; lower mass cutoff {mass["lower"]:.2f} px; no upper cutoff. {len(interior)} retained.')
    save(nodes_plot(interior,'Crossing nodes after lower mass filter',rejected=small),'mass_filtered_nodes','White: retained geometric component centers. Red: small regions rejected by lower mass cutoff.')
    preliminary,bootstrap=scan_cleanup(interior,image.size,width)
    save(scans_plot(bootstrap),'bootstrap_maxima','Bootstrap maxima cleanup provides the straight-line scaffold for completing edge endpoints.')
    grid={a:r[3].tolist() for a,r in bootstrap.items()}
    edge_scores,proposals,edge_threshold,depth,_=detect_edges(scalar,width,threshold_method='histogram')
    cross_profiles={'top':cross[:depth].max(axis=0),'bottom':cross[-depth:].max(axis=0),'left':cross[:,:depth].max(axis=1),'right':cross[:,-depth:].max(axis=1)}
    endpoints=[];unsupported=[]
    for side in ('top','bottom','left','right'):
        lines=grid['x'] if side in ('top','bottom') else grid['y']
        records=complete_endpoints(lines,[n for n in proposals if n['side']==side],edge_scores[side],cross_profiles[side],allow_missing=True)
        for r in records:
            if r['origin']=='unsupported':
                unsupported.append(dict(r,side=side));continue
            p=r['profile_center'];x,y=(p,0) if side=='top' else (p,image.height-1) if side=='bottom' else (0,p) if side=='left' else (image.width-1,p)
            endpoints.append(dict(r,x=float(x),y=float(y),kind='edge',side=side,scan_axes=['x'] if side in ('top','bottom') else ['y']))
    for j,n in enumerate(endpoints):n['id']=len(all_nodes)+j+1
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    for ax,(side,values) in zip(axes.ravel(),edge_scores.items()):
        ax.plot(values);ax.axhline(edge_threshold,color='gray',ls='--')
        for n in endpoints:
            if n['side']==side:ax.axvline(n['profile_center'],color='orange',alpha=.4)
        ax.set(title=side,xlabel='Edge position (px)',ylabel='Width-matched score')
    save(fig,'edge_profiles',f'Histogram edge threshold {edge_threshold:.4f}; {len(proposals)} proposals; {len(endpoints)} endpoints; {len(unsupported)} unsupported endpoints recorded. No mass cutoff for endpoints.')
    combined=interior+endpoints
    save(nodes_plot(combined,'Crossing nodes and completed edge nodes',grid),'edge_nodes_and_straight_grid','Yellow: accepted/completed endpoints. Cyan/magenta: preliminary straight grid. White: crossing nodes.')
    cleaned,scans=scan_cleanup(combined,image.size,width)
    save(scans_plot(scans),'final_maxima_cleanup',f'Endpoints are added before final maxima cleanup. {len(combined)} nodes become {len(cleaned)}. Gray/blue: local spacing before/after cleanup.')
    save(nodes_plot(cleaned,'Nodes after maxima cleanup'),'post_maxima_nodes','No line-score or isolated-neighbor rejection is used.')
    group_results={a:group_deviations(cleaned,a,extent,width) for a,extent in zip(('x','y'),image.size)}
    deviations=[r for result in group_results.values() for r in result[-1]]
    reject_ids=local_outlier_ids(deviations,2)
    final=[n for n in cleaned if n['id'] not in reject_ids];removed=[n for n in cleaned if n['id'] in reject_ids]
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,(a,result) in zip(axes,group_results.items()):
        for boundary in result[2]:ax.axvline(boundary,color='gray',ls=':',alpha=.4)
        r=[r for r in result[-1] if r['distance_in_sd'] is not None]
        ax.scatter([v['coordinate'] for v in r],[v['distance_in_sd'] for v in r],s=20)
        ax.axhline(2,color='red',ls='--');ax.set(title=f'{a}: {len(result[3])} minima-defined groups',xlabel=f'{a} coordinate',ylabel='Absolute within-group SD')
    save(fig,'local_group_outliers','One-pass population statistics from post-maxima groups. Reject a node if either eligible coordinate is at least 2 SD from its own group mean; undefined SD does not reject.')
    save(nodes_plot(final,'Nodes after the local 2-SD filter',rejected=removed),'two_sd_filtered_nodes',f'{len(removed)} rejected nodes marked red; {len(final)} retained. Statistics are not recomputed after removal.')
    points=np.array([[n['x'],n['y']] for n in final]);curves=[];ignored=[]
    for dim,a,family in (((0,'x','vertical'),(1,'y','horizontal')) if fit_curves else ()):
        peaks=scans[a][3];eligible=np.array([a in n.get('scan_axes',['x','y']) for n in final]);assignment=np.argmin(abs(points[:,dim,None]-peaks),axis=1)
        for j,n in enumerate(final):
            if eligible[j] and n.get('kind')=='edge' and n.get('line_index') is not None:
                # Match to the nearest retained final maximum; a removed bootstrap line cannot shift identities.
                original_position=grid[a][n['line_index']];assignment[j]=np.argmin(abs(peaks-original_position))
        for i,peak in enumerate(peaks):
            ids=np.flatnonzero(eligible & (assignment==i));along=points[ids,1-dim]
            if len(np.unique(along))<4:
                ignored.append({'family':family,'peak':int(peak),'nodes':len(ids)});continue
            controls,rmse=fit_curve(along,points[ids,dim],image.size[1-dim],image.size[dim],peak,smoothness=0,robust_scale=width)
            cp=np.empty((4,2));cp[:,1-dim]=np.linspace(0,image.size[1-dim]-1,4);cp[:,dim]=controls
            curves.append({'family':family,'peak':int(peak),'control_points':cp.tolist(),'rmse':rmse,'node_ids':[final[j]['id'] for j in ids]})
    if fit_curves:
        fig=nodes_plot(final,'Final cubic Bezier curves: four controls, no bending penalty');ax=fig.axes[0]
        for c in curves:
            p=bezier_basis(np.linspace(0,1,500))@np.array(c['control_points']);ax.plot(p[:,0],p[:,1],color='magenta' if c['family']=='vertical' else 'cyan',lw=1.1)
        save(fig,'final_bezier_fit',f'{len(curves)} curves. {len(ignored)} undersupported curves omitted. Robust node residual loss; all four transverse controls fitted together per curve.')
    accepted_ids={n['id'] for n in final}
    candidates=[dict(n,kind=n.get('kind','interior'),spatial_accepted=n['id'] in accepted_ids)
                for n in all_nodes+endpoints]
    result={'source':str(Path(source).resolve()),'frame':index,'width':width,'width_profiles':len(measurements),
        'threshold':threshold,'threshold_sustained':selection['sustained'],
        'detection_issues':[] if selection['sustained'] else ['Crossing threshold uses the shallowest slope without a sustained interval'],
        'seed':selected,'color_model':color_model,'components':len(all_nodes),'mass_kept':len(interior),
        'edge_proposals':len(proposals),'edge_threshold':edge_threshold,'unsupported_endpoints':unsupported,'edge_nodes':len(endpoints),'post_maxima':len(cleaned),'removed_2sd':len(removed),
        'final_nodes':final,'candidate_nodes':candidates,'image_size':list(image.size),'scaffold':grid,'fit_curves':fit_curves,'diagnostics':diagnostics,
        'outlier_nodes':removed,'group_statistics':{a:{'minima':r[2].tolist(),'groups':r[3],'deviations':r[4]} for a,r in group_results.items()},
        'maxima':{a:{'raw':r[2].tolist(),'clean':r[3].tolist(),'removals':r[4]} for a,r in scans.items()},
        'curves':curves,'ignored_curves':ignored,'artifacts':artifacts,'single_thread_env':{n:os.environ[n] for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
