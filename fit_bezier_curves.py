"""One cubic Bezier per cleaned scan peak, with four transverse parameters."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy.optimize import least_squares


def bezier_basis(t):
    t = np.asarray(t, dtype=float)
    return np.column_stack(((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3))


def fit_curve(along, across, extent, transverse_extent, initial, smoothness=0, robust_scale=8):
    along, across = np.asarray(along), np.asarray(across)
    if not len(along):
        raise ValueError("A curve has no assigned nodes")
    if smoothness < 0 or robust_scale <= 0:
        raise ValueError("Require nonnegative smoothness and positive robust residual scale")
    required = 4 if smoothness == 0 else 2
    if len(np.unique(along)) < required:
        raise ValueError(f"This curve requires at least {required} distinct longitudinal node positions")
    basis = bezier_basis(along/(extent-1))
    difference = np.array([[1.,-2,1,0], [0,1,-2,1]])
    penalty = np.sqrt(smoothness/2)*difference

    def residual(controls):
        r = basis@controls-across
        root = np.sqrt(1+(r/robust_scale)**2)
        # This squared residual is exactly the soft-L1 data loss, without affecting the bend penalty.
        return np.r_[r*np.sqrt(2/(root+1)/len(across)), penalty@controls]

    def jacobian(controls):
        r = basis@controls-across
        root = np.sqrt(1+(r/robust_scale)**2)
        factor = np.sqrt((root+1)/(2*len(across)))/root
        return np.vstack((factor[:,None]*basis, penalty))

    result = least_squares(residual,np.full(4,float(initial)),jac=jacobian,
                           bounds=(np.zeros(4),np.full(4,transverse_extent-1)),
                           max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-8)
    if not result.success:
        raise RuntimeError(f"Bezier fit failed: {result.message}")
    return result.x, float(np.sqrt(np.mean((basis@result.x-across)**2)))


def node_available(node):
    return node.get('measured_xy') is not None or (node['kind'] == 'edge' and node.get('status') == 'held')


def curve_intersections(vertical, horizontal, size):
    """All crossings of the fitted graphs x(y) and y(x), inside the image."""
    from numpy.polynomial import Polynomial
    extent = np.asarray(size, dtype=float)-1
    if extent.shape != (2,) or np.any(extent <= 0):
        raise ValueError('Curve intersections require a positive image extent')
    polynomials = []
    for curve, transverse in ((vertical, 0), (horizontal, 1)):
        points = np.asarray(curve['control_points'], dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError('Expected four finite 2D Bezier controls')
        if not np.allclose(points[:, 1-transverse], np.linspace(0, extent[1-transverse], 4), rtol=0, atol=1e-7):
            raise ValueError('Crossing calculation requires the fitted evenly spaced longitudinal controls')
        a, b, c, d = points[:, transverse]/extent[transverse]
        polynomials.append(Polynomial([a, 3*(b-a), 3*(a-2*b+c), -a+3*b-3*c+d]))
    x_of_y, y_of_x = polynomials
    equation = (x_of_y(y_of_x)-Polynomial([0., 1.])).trim(tol=1e-12)
    if equation.degree() == 0 and abs(equation.coef[0]) < 1e-12:
        raise ValueError('Coincident curves do not define a unique grid crossing')
    crossings = []
    for root in equation.roots():
        if abs(root.imag) > 1e-7 or not -1e-8 <= root.real <= 1+1e-8:
            continue
        x = float(np.clip(root.real, 0, 1))
        y = float(y_of_x(x))
        if not -1e-8 <= y <= 1+1e-8:
            continue
        point = np.array([x, np.clip(y, 0, 1)])*extent
        residual = abs(x-float(x_of_y(y)))*extent[0]
        if residual <= 1e-5 and not any(np.linalg.norm(point-p) < 1e-4 for p in crossings):
            crossings.append(point)
    return np.array(crossings).reshape(-1, 2)


def fit_labeled_curves(nodes, size, width):
    """Fit observations and held boundary endpoints using persistent labels."""
    curves, ignored = [], []
    measured = [n for n in nodes if node_available(n)]
    for dim, label, family in ((0, 'column', 'vertical'), (1, 'row', 'horizontal')):
        for identity in sorted({n[label] for n in measured if n[label] is not None}):
            selected = [n for n in measured if n[label] == identity]
            points = np.array([[n['x'], n['y']] for n in selected])
            if len(np.unique(points[:, 1-dim])) < 4:
                ignored.append(dict(family=family, line_id=identity, node_ids=[n['id'] for n in selected], reason='fewer than four distinct longitudinal positions'))
                continue
            controls, rmse = fit_curve(points[:, 1-dim], points[:, dim], size[1-dim], size[dim],
                                       np.median(points[:, dim]), smoothness=0, robust_scale=width)
            cp = np.empty((4, 2))
            cp[:, 1-dim], cp[:, dim] = np.linspace(0, size[1-dim]-1, 4), controls
            curves.append(dict(family=family, line_id=identity, control_points=cp.tolist(),
                               rmse=rmse, node_ids=[n['id'] for n in selected]))
    return curves, ignored


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",type=Path,default=Path("output/cleaned_scan_peaks/analysis.json"))
    parser.add_argument("--output",type=Path,default=Path("output/bezier_no_bending"))
    parser.add_argument("--smoothness",type=float,default=0)
    args=parser.parse_args()
    data=json.loads(args.analysis.read_text())
    nodes=data['nodes'];points=np.array([[n['x'],n['y']] for n in nodes])
    with Image.open(data['source']) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
    lengths=np.array(image.size)
    curves, ignored=[],[]
    for across_axis,name,label in ((0,'x','vertical'),(1,'y','horizontal')):
        peaks=np.array(data['scans'][name]['retained_peaks'])
        eligible=np.array([name in node.get('scan_axes',['x','y']) for node in nodes])
        assignment=np.argmin(abs(points[:,across_axis,None]-peaks),axis=1)
        for j,node in enumerate(nodes):
            if eligible[j] and node.get('kind')=='edge' and node.get('line_index') is not None:
                index=node['line_index']
                if not 0<=index<len(peaks):
                    raise ValueError('Edge endpoint line identity does not match the cleaned grid')
                assignment[j]=index
        along_axis=1-across_axis
        for i,peak in enumerate(peaks):
            selected=np.flatnonzero(eligible & (assignment==i))
            required = 4 if args.smoothness == 0 else 2
            if len(np.unique(points[selected,along_axis])) < required:
                ignored.append({'family':label,'peak':int(peak),'node_ids':[nodes[j]['id'] for j in selected],
                                'reason':f'fewer than {required} distinct longitudinal node positions'})
                continue
            controls,rmse=fit_curve(points[selected,along_axis],points[selected,across_axis],
                                   lengths[along_axis],lengths[across_axis],peak,args.smoothness,data['sigma_pixels'])
            control_points=np.empty((4,2))
            control_points[:,along_axis]=np.linspace(0,lengths[along_axis]-1,4)
            control_points[:,across_axis]=controls
            curves.append({'family':label,'peak':int(peak),'control_points':control_points.tolist(),
                           'node_ids':[nodes[j]['id'] for j in selected],'rmse_pixels':rmse,
                           'fit_mode':'four_transverse_controls'})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'curves.json').write_text(json.dumps({'source':data['source'],'smoothness':args.smoothness,
        'robust_scale_pixels':data['sigma_pixels'],'control_points_per_curve':4,
        'fixed_longitudinal_control_positions':'0, 1/3, 2/3, 1 of image extent',
        'assignment':'direction-eligible nodes only; edge endpoints use known line identity, other nodes use nearest cleaned maximum',
        'smoothness_penalty':'mean squared second differences of transverse control coordinates',
        'curves':curves,'ignored':ignored},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'text.usetex':False,'font.family':'DejaVu Sans'})
    basis=bezier_basis(np.linspace(0,1,600))
    fig,axis=plt.subplots(figsize=(13,7),constrained_layout=True)
    axis.imshow(ImageEnhance.Brightness(image).enhance(3))
    for family,color in (('vertical','magenta'),('horizontal','cyan')):
        first=True
        for curve in curves:
            if curve['family']==family:
                line=basis@np.array(curve['control_points'])
                axis.plot(line[:,0],line[:,1],color=color,lw=1.2,label=family if first else None)
                first=False
    axis.scatter(points[:,0],points[:,1],s=22,marker='+',color='white',label=f'{len(nodes)} detected nodes')
    axis.set_title(f"{sum(c['family']=='vertical' for c in curves)} vertical + {sum(c['family']=='horizontal' for c in curves)} horizontal cubic Bezier curves; four controls each; bending penalty {args.smoothness:g}")
    axis.set_axis_off();axis.legend(facecolor='#222222',labelcolor='white',loc='upper left')
    fig.savefig(args.output/'curves.png',dpi=180);fig.savefig(args.output/'curves.pdf');plt.close(fig)
    print(json.dumps({'curves':len(curves),'vertical':sum(c['family']=='vertical' for c in curves),
                      'horizontal':sum(c['family']=='horizontal' for c in curves),'ignored':ignored,
                      'rmse_range_pixels':[min(c['rmse_pixels'] for c in curves),max(c['rmse_pixels'] for c in curves)]}))


if __name__=='__main__':
    main()
