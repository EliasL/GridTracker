"""Detect string endpoints on the four image borders."""

import numpy as np
from scipy.ndimage import correlate1d

from crossing_nodes import component_nodes


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


def detect_edges(field, width):
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
    threshold = histogram_threshold(np.concatenate(list(scores.values())))
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
    return scores, nodes, threshold, depth
