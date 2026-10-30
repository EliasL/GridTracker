import unittest

import numpy as np

from crossing_nodes import filter_by_mass
from experiments.fit_point_grid import fourier_spacing, fit_offset, remove_close_points, wrapped_residual


class PointGridTests(unittest.TestCase):
    def test_fourier_fit_with_missing_jittered_crossings(self):
        rng = np.random.default_rng(19)
        x, y = np.meshgrid(17+91*np.arange(9), 29+73*np.arange(7))
        points = np.column_stack((x.ravel(),y.ravel())) + rng.normal(0, .8, (x.size,2))
        points = points[rng.random(len(points)) > .25]
        for axis, (pitch, offset, extent) in enumerate(((91,17,1000), (73,29,700))):
            found, _ = fourier_spacing(points[:,axis], extent)
            self.assertAlmostEqual(found, pitch, delta=.3)
            phase = fit_offset(points[:,axis], found)
            self.assertLess(abs(wrapped_residual(offset, found, phase)), 1)

    def test_grid_nearest_wins_among_close_points(self):
        points = np.array([[12.,12.], [10.,10.], [110.,10.]])
        distances, pairs, crowded, kept, removed = remove_close_points(points, np.array([100.,100.]), np.array([10.,10.]))
        self.assertEqual(pairs, [(0,1)])
        self.assertEqual(crowded, {0,1})
        self.assertEqual(kept, {1,2})
        self.assertEqual(removed, {0:1})
        self.assertGreater(distances[0], distances[1])

    def test_close_chain_does_not_remove_separated_endpoints(self):
        points = np.array([[0.,0.], [40.,0.], [80.,0.]])
        _, _, _, kept, removed = remove_close_points(points, np.array([100.,100.]), np.array([0.,0.]))
        self.assertEqual(kept, {0,2})
        self.assertEqual(removed, {1:0})

    def test_lower_mass_filter_keeps_large_regions(self):
        nodes = [{"pixels": value} for value in (1,10,10,10,19)]
        kept, removed, stats = filter_by_mass(nodes, upper_limit=False)
        self.assertEqual([n['pixels'] for n in kept], [10,10,10,19])
        self.assertEqual([n['pixels'] for n in removed], [1])
        self.assertIsNone(stats['upper'])


if __name__ == '__main__':
    unittest.main()
