import unittest

import numpy as np

from crossing_nodes import choose_threshold, component_nodes, threshold_coverage, filter_by_mass


class ThresholdTests(unittest.TestCase):
    def test_coverage_excludes_zero_and_uses_strict_threshold(self):
        t, fraction = threshold_coverage(np.array([[0, .1, .1], [.2, .4, 0]]))
        np.testing.assert_allclose(t, [.1, .2])
        np.testing.assert_allclose(fraction, [2/6, 1/6])

    def test_detects_known_start_of_shallow_log_slope(self):
        u = (np.arange(50000) + .5) / 50000
        z = -np.log10(u)
        values = np.where(z <= 1, z/10,
                          np.where(z <= 1.525, .1+(z-1)/1.5, .45+(z-1.525)/10))
        result = choose_threshold(values.reshape(200,250))
        self.assertAlmostEqual(result['threshold'], .1, delta=.04)
        self.assertGreater(result['plateau_end'], .3)

    def test_geometric_centroids_use_eight_connectivity(self):
        scores = np.zeros((4,5))
        scores[0,0], scores[1,1], scores[3,4] = .3, .7, .5
        _, nodes = component_nodes(scores, .2)
        self.assertEqual(len(nodes), 2)
        self.assertEqual((nodes[0]['x'], nodes[0]['y']), (.5,.5))
        self.assertAlmostEqual(nodes[0]['score_mass'], 1.0)
        self.assertAlmostEqual(nodes[1]['score_mass'], .5)

    def test_mass_filter_is_one_pass_and_includes_bounds(self):
        nodes = [{"pixels": value} for value in (1, 1, 3, 3)]
        kept, rejected, stats = filter_by_mass(nodes, mass_key="pixels")
        self.assertEqual(stats["mean"], 2)
        self.assertEqual(stats["std"], 1)
        self.assertEqual(kept, nodes)
        self.assertEqual(rejected, [])
        kept, rejected, stats = filter_by_mass([{"pixels": value} for value in (1, 10, 10, 10, 19)], mass_key="pixels")
        self.assertEqual(len(kept), 3)
        self.assertEqual([node["pixels"] for node in rejected], [1,19])

    def test_weighted_mass_uses_score_sum_rather_than_pixel_count(self):
        nodes = [{"pixels": 100, "score_mass": value} for value in (1,10,10,10,19)]
        kept, rejected, stats = filter_by_mass(nodes, mass_key="score_mass")
        self.assertEqual(len(kept), 3)
        self.assertEqual([n["score_mass"] for n in rejected], [1,19])
        self.assertEqual(stats["mass_key"], "score_mass")


if __name__ == '__main__':
    unittest.main()
