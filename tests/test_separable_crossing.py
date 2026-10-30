import unittest

import numpy as np
from scipy.ndimage import correlate,gaussian_filter
from detection import crossing_evidence


class SeparableCrossingTests(unittest.TestCase):
    def test_equivalent_to_full_two_dimensional_correlations(self):
        field=np.random.default_rng(7).uniform(0,1,(55,63));width=4
        radius=int(np.ceil(3*width));gap=int(np.ceil(width))
        yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
        masks=[(abs(yy)<=width/2)&(xx>=gap),(abs(yy)<=width/2)&(xx<=-gap),
               (abs(xx)<=width/2)&(yy>=gap),(abs(xx)<=width/2)&(yy<=-gap)]
        responses=[correlate(field,m.astype(float)/m.sum(),mode='nearest') for m in masks]
        off=(abs(xx)>=gap)&(abs(yy)>=gap)
        background=correlate(field,off.astype(float)/off.sum(),mode='nearest')
        expected=gaussian_filter(np.clip((np.minimum.reduce(responses)-background)/np.maximum(1-background,1e-6),0,1),.6)
        np.testing.assert_allclose(crossing_evidence(field,width),expected,atol=1e-12)
