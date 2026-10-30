import unittest

import numpy as np

from edge_nodes import detect_edges, edge_line_score


class EdgeTests(unittest.TestCase):
    def test_flat_color_does_not_propose_a_string(self):
        score = edge_line_score(np.ones(100), 8)
        self.assertLess(score.max(), 1e-6)

    def test_centered_string_response_is_centered(self):
        x = np.arange(100)
        profile = np.exp(-.5*((x-40)/(8/2.35482))**2)
        score = edge_line_score(profile, 8)
        self.assertEqual(score.argmax(), 40)
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(((score>=0)&(score<=1)).all())

    def test_proposed_nodes_lie_on_correct_borders(self):
        yy, xx = np.indices((120,160))
        field = np.zeros((120,160))
        for x in (20,60,100,140):
            field = np.maximum(field,np.exp(-.5*((xx-x)/(4/2.35482))**2))
        for y in (20,60,100):
            field = np.maximum(field,np.exp(-.5*((yy-y)/(4/2.35482))**2))
        _, nodes, _, _ = detect_edges(field,4)
        self.assertEqual(len(nodes),14)
        for node in nodes:
            if node['side'] in ('top','bottom'):
                self.assertIn(node['y'],[0,119])
                self.assertLess(min(abs(node['x']-x) for x in (20,60,100,140)),1)
            else:
                self.assertIn(node['x'],[0,159])
                self.assertLess(min(abs(node['y']-y) for y in (20,60,100)),1)


if __name__=='__main__':
    unittest.main()
