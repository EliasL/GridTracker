from copy import deepcopy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from fit_bezier_curves import curve_intersections
from review_session import curve_crossing_snapshot, export_report


def curve(family, transverse, identity=0):
    points = np.empty((4, 2))
    dim = 0 if family == 'vertical' else 1
    points[:, dim] = transverse
    points[:, 1-dim] = np.linspace(0, 100, 4)
    return dict(family=family, line_id=identity, control_points=points.tolist(), rmse=0., node_ids=[999])


def example_report():
    return dict(fps=10., frames=[dict(frame=i, source='example.png', image_size=[101, 101], human_verified=i == 0,
        nodes=[dict(id=999, kind='interior', row=0, column=0, x=5., y=7., measured_xy=[5., 7.])],
        curves=[curve('vertical', [25+2*i]*4), curve('horizontal', [40+3*i]*4)], ignored_curves=[])
        for i in range(3)])


class CurveCrossingTests(unittest.TestCase):
    def test_straight_curved_and_boundary_crossings(self):
        np.testing.assert_allclose(curve_intersections(curve('vertical', [25]*4), curve('horizontal', [40]*4), [101, 101]), [[25, 40]])
        vertical = curve('vertical', [20, 20+10/3, 20+20/3+5/3, 35])
        np.testing.assert_allclose(curve_intersections(vertical, curve('horizontal', [50]*4), [101, 101]), [[26.25, 50]], atol=1e-7)
        np.testing.assert_allclose(curve_intersections(curve('vertical', [0]*4), curve('horizontal', [100]*4), [101, 101]), [[0, 100]])

    def test_multiple_crossings_are_reported_as_ambiguous(self):
        vertical = curve('vertical', np.linspace(0, 100, 4))
        horizontal = curve('horizontal', 100*np.array([.08, .08+.34/3, .08+2*.34/3+.5, .92]))
        points = curve_intersections(vertical, horizontal, [101, 101])
        np.testing.assert_allclose(points, [[20, 20], [50, 50], [80, 80]], atol=1e-5)
        report = example_report()
        report['frames'][0]['curves'] = [vertical, horizontal]
        crossing = curve_crossing_snapshot(report)['frames'][0]['nodes'][0]
        self.assertEqual(crossing['status'], 'ambiguous_intersections')
        self.assertEqual(crossing['intersection_count'], 3)
        self.assertIsNone(crossing['x'])

    def test_export_uses_curve_crossings_and_their_motion_not_editable_guides(self):
        report = example_report()
        original = json.dumps(report, sort_keys=True)
        with TemporaryDirectory() as directory:
            results = [dict(frame=i, candidate_nodes=[]) for i in range(3)]
            paths = export_report(report, results, Path(directory))
            with paths[0].open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]['id'], 'r0:c0')
            self.assertAlmostEqual(float(rows[1]['x']), 27.)
            self.assertAlmostEqual(float(rows[1]['y']), 43.)
            self.assertAlmostEqual(float(rows[1]['velocity_x']), 20.)
            self.assertAlmostEqual(float(rows[1]['velocity_y']), 30.)
            self.assertAlmostEqual(float(rows[1]['acceleration_x']), 0.)
            self.assertEqual(rows[1]['measurement_source'], 'bezier_intersection')
        self.assertEqual(json.dumps(report, sort_keys=True), original)

    def test_missing_curve_does_not_fall_back_to_guide_positions(self):
        report = example_report()
        report['frames'][1]['curves'] = report['frames'][1]['curves'][:1]
        snapshot = curve_crossing_snapshot(report)
        self.assertEqual(snapshot['frames'][1]['nodes'][0]['status'], 'missing_curve')
        self.assertIsNone(snapshot['frames'][1]['nodes'][0]['x'])
        self.assertTrue(all(f['nodes'][0]['velocity'] is None for f in snapshot['frames']))

    def test_coincident_or_malformed_graphs_raise(self):
        with self.assertRaisesRegex(ValueError, 'Coincident'):
            curve_intersections(curve('vertical', np.linspace(0, 100, 4)), curve('horizontal', np.linspace(0, 100, 4)), [101, 101])
        malformed = curve('vertical', [25]*4)
        malformed['control_points'][1][1] += 2
        with self.assertRaisesRegex(ValueError, 'longitudinal'):
            curve_intersections(malformed, curve('horizontal', [40]*4), [101, 101])


if __name__ == '__main__':
    unittest.main()
