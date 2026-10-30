"""Fit cubic Bezier strings and derive their grid intersections."""

import numpy as np
from scipy.optimize import least_squares


def bezier_basis(t):
    t = np.asarray(t, dtype=float)
    return np.column_stack(((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3))


def fit_curve(along, across, extent, transverse_extent, initial, robust_scale=8):
    along, across = np.asarray(along), np.asarray(across)
    if not len(along):
        raise ValueError("A curve has no assigned nodes")
    if robust_scale <= 0:
        raise ValueError("Require a positive robust residual scale")
    if len(np.unique(along)) < 4:
        raise ValueError("This curve requires at least four distinct longitudinal node positions")
    basis = bezier_basis(along/(extent-1))

    def residual(controls):
        r = basis@controls-across
        root = np.sqrt(1+(r/robust_scale)**2)
        # This squared residual is exactly the soft-L1 data loss.
        return r*np.sqrt(2/(root+1)/len(across))

    def jacobian(controls):
        r = basis@controls-across
        root = np.sqrt(1+(r/robust_scale)**2)
        factor = np.sqrt((root+1)/(2*len(across)))/root
        return factor[:,None]*basis

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


def fit_node_curve(nodes, size, width, dim, initial):
    points = np.array([[n['x'], n['y']] for n in nodes])
    controls, rmse = fit_curve(points[:, 1-dim], points[:, dim], size[1-dim], size[dim],
                               initial, robust_scale=width)
    cp = np.empty((4, 2))
    cp[:, 1-dim], cp[:, dim] = np.linspace(0, size[1-dim]-1, 4), controls
    return dict(control_points=cp.tolist(), rmse=rmse, node_ids=[n['id'] for n in nodes])


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
            curve = fit_node_curve(selected, size, width, dim, np.median(points[:, dim]))
            curves.append(dict(curve, family=family, line_id=identity))
    return curves, ignored
