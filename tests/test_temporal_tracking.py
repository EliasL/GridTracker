import unittest

import cv2
import numpy as np

from fit_bezier_curves import fit_labeled_curves
from temporal_tracking import NodeTracker, TrackingConfig, assign, topology_conflicts


def scene(shift=(0, 0), hidden=(), extra=()):
    y, x = np.mgrid[:180, :180]
    image = np.zeros((180, 180), dtype=float)
    nodes = []
    for row, py in enumerate((40, 90, 140)):
        for column, px in enumerate((40, 90, 140)):
            identity = row*3+column
            px, py0 = px+shift[0], py+shift[1]
            nodes.append(dict(id=identity, x=float(px), y=float(py0), row=row, column=column,
                              kind='interior', scan_axes=['x', 'y'], spatial_accepted=True))
            if identity not in hidden:
                image += np.exp(-.5*((x-px)/1.5)**2-.5*((y-py0)/10)**2)
                image += np.exp(-.5*((y-py0)/1.5)**2-.5*((x-px)/10)**2)
    for px, py in extra:
        image += np.exp(-.5*((x-px)/1.5)**2-.5*((y-py)/10)**2)
        image += np.exp(-.5*((y-py)/1.5)**2-.5*((x-px)/10)**2)
    return np.uint8(np.clip(image*120, 0, 255)), nodes


class TrackingTests(unittest.TestCase):
    def tracker(self, **kwargs):
        image, nodes = scene()
        return NodeTracker(nodes, image, width=4, spacing=50, **kwargs)

    def test_assignment_can_leave_every_track_unmatched(self):
        self.assertEqual(assign(np.full((3, 2), np.inf)), {})
        self.assertEqual(assign(np.empty((3, 0))), {})
        self.assertEqual(assign([[.1], [.2]]), {0: 0})

    def test_translation_preserves_ids_and_estimates_motion(self):
        tracker = self.tracker()
        for step in (1, 2):
            image, nodes = scene((2*step, step))
            records, _ = tracker.step(image, nodes)
            self.assertEqual([n['id'] for n in records], list(range(9)))
            self.assertTrue(all(n['status'] == 'observed' for n in records))
            np.testing.assert_allclose([[n['x'], n['y']] for n in records], [[n['x'], n['y']] for n in nodes])
        self.assertTrue(all(n['acceleration'] is not None for n in records))
        np.testing.assert_allclose(records[4]['measurement_velocity'], [2, 1])

    def test_optical_flow_can_recover_a_detection_that_was_not_proposed(self):
        tracker = self.tracker()
        image, nodes = scene((2, 1))
        records, _ = tracker.step(image, [])
        self.assertTrue(all(n['measurement_source'] == 'optical_flow' for n in records))
        np.testing.assert_allclose([[n['x'], n['y']] for n in records], [[n['x'], n['y']] for n in nodes], atol=.2)

    def test_occlusion_becomes_lost_then_recovers_same_identity(self):
        tracker = self.tracker()
        for step in range(1, 5):
            image, nodes = scene(hidden=(4,))
            records, _ = tracker.step(image, [n for n in nodes if n['id'] != 4])
            missing = records[4]
            self.assertEqual(missing['status'], 'predicted' if step <= 3 else 'lost')
            self.assertIsNone(missing['measured_xy'])
            self.assertIsNone(missing['velocity'])
            self.assertIsNone(missing['acceleration'])
        image, nodes = scene()
        records, _ = tracker.step(image, nodes)
        self.assertEqual(records[4]['status'], 'recovered')
        self.assertEqual(records[4]['id'], 4)
        self.assertIsNone(records[4]['acceleration'])
        self.assertEqual(len(tracker.tracks), 9)

    def test_moving_node_stops_at_its_last_observation_when_lost(self):
        tracker = self.tracker(config=TrackingConfig(use_flow=False))
        for step in (1, 2):
            image, nodes = scene((2*step, step))
            records, _ = tracker.step(image, nodes)
        last = np.array(records[4]['measured_xy'])
        self.assertGreater(np.linalg.norm(records[4]['velocity']), 0)
        for step in range(3, 8):
            image, nodes = scene((2*step, step), hidden=(4,))
            records, _ = tracker.step(image, [n for n in nodes if n['id'] != 4])
            np.testing.assert_array_equal([records[4]['x'], records[4]['y']], last)
            np.testing.assert_array_equal(tracker.tracks[4]['mean'][2:], [0, 0])
            if step > 3:
                np.testing.assert_array_equal(records[4]['prediction_xy'], last)
        image, nodes = scene((16, 8))
        records, _ = tracker.step(image, nodes)
        self.assertEqual(records[4]['status'], 'recovered')
        np.testing.assert_array_equal(records[4]['measured_xy'], [106, 98])

    def test_edge_is_held_indefinitely_and_updates_only_on_identification(self):
        def edge_image(position=None):
            image, nodes = scene()
            if position is not None:
                y, x = np.mgrid[:180, :180]
                ribbon = 120*np.exp(-.5*((x-position)/1.5)**2-.5*(y/10)**2)
                image = np.uint8(np.clip(image.astype(float)+ribbon, 0, 255))
            return image, nodes
        image, nodes = edge_image(40)
        edge = dict(id=100, x=40., y=0., kind='edge', side='top', row=None, column=0, scan_axes=['x'])
        tracker = NodeTracker(nodes+[edge], image, width=4, spacing=50, config=TrackingConfig(use_flow=False))
        image, detections = edge_image(42)
        records, _ = tracker.step(image, detections+[dict(edge, id=500, x=42.)])
        self.assertEqual(records[-1]['status'], 'observed')
        for _ in range(10):
            image, detections = edge_image()
            records, _ = tracker.step(image, detections)
            held = records[-1]
            self.assertEqual(held['status'], 'held')
            self.assertEqual((held['id'], held['x'], held['y']), (100, 42., 0.))
            self.assertIsNone(held['measured_xy'])
            self.assertIsNone(held['velocity'])
        curves, _ = fit_labeled_curves(records, (180, 180), 4)
        self.assertTrue(any(100 in c['node_ids'] for c in curves))
        image, detections = edge_image(45)
        records, _ = tracker.step(image, detections+[dict(edge, id=501, x=45.)])
        self.assertEqual((records[-1]['status'], records[-1]['x']), ('recovered', 45.))

    def test_spurious_persistent_detection_never_becomes_verified(self):
        tracker = self.tracker()
        for step in (1, 2):
            image, nodes = scene(extra=((65, 65),))
            nodes.append(dict(id=100, x=65., y=65., kind='interior', spatial_accepted=True))
            records, unclaimed = tracker.step(image, nodes)
            self.assertEqual(len(records), 9)
            candidate = next(n for n in unclaimed if n['id'] == 100)
            self.assertEqual(candidate['reason'], 'unverified_new')
            self.assertEqual(candidate['candidate_id'], 'candidate-0')
            self.assertEqual(candidate['candidate_hits'], step)
        image, nodes = scene()
        for _ in range(3):
            tracker.step(image, nodes)
        self.assertFalse(tracker.candidates)

    def test_ambiguous_detections_do_not_force_a_match(self):
        tracker = self.tracker()
        image, nodes = scene()
        nodes = [n for n in nodes if n['id'] != 4]+[
            dict(id=90, x=89., y=90., kind='interior', spatial_accepted=True),
            dict(id=91, x=91., y=90., kind='interior', spatial_accepted=True)]
        records, _ = tracker.step(image, nodes)
        self.assertEqual(records[4]['reason'], 'ambiguous_detections')
        self.assertIsNone(records[4]['measured_xy'])

    def test_spatially_rejected_candidate_can_be_rescued(self):
        tracker = self.tracker()
        image, nodes = scene((1, 0))
        nodes[4]['spatial_accepted'] = False
        records, _ = tracker.step(image, nodes)
        self.assertEqual(records[4]['detection_id'], 4)
        self.assertEqual(records[4]['status'], 'observed')

    def test_grid_order_conflicts_reject_both_crossings(self):
        _, nodes = scene()
        proposals = {0: dict(xy=np.array([95, 40])), 1: dict(xy=np.array([35, 40]))}
        self.assertEqual(topology_conflicts(nodes, proposals, 4), {0, 1})

    def test_motion_units_follow_frame_interval(self):
        tracker = self.tracker(dt=.1, config=TrackingConfig(use_flow=False))
        image, nodes = scene((2, 1))
        records, _ = tracker.step(image, nodes)
        np.testing.assert_allclose(records[4]['measurement_velocity'], [20, 10])

    def test_variable_frame_intervals_and_an_occlusion_use_elapsed_time(self):
        tracker = self.tracker(dt=.1, config=TrackingConfig(use_flow=False))
        image, nodes = scene((2, 1))
        tracker.step(image, nodes, dt=.1)
        image, nodes = scene((6, 3), hidden=(4,))
        tracker.step(image, [n for n in nodes if n['id'] != 4], dt=.2)
        image, nodes = scene((8, 4))
        records, _ = tracker.step(image, nodes, dt=.1)
        self.assertEqual(records[4]['status'], 'recovered')
        np.testing.assert_allclose(records[4]['measurement_velocity'], [20, 10])

    def test_missing_predictions_cannot_support_a_curve(self):
        nodes = [dict(id=i, x=20., y=float(i*20), row=i, column=0, kind='interior',
                      measured_xy=[20., float(i*20)] if i < 3 else None) for i in range(5)]
        curves, ignored = fit_labeled_curves(nodes, (101, 101), 4)
        self.assertFalse(curves)
        self.assertTrue(any(c['family'] == 'vertical' for c in ignored))

    def test_invalid_input_raises(self):
        tracker = self.tracker()
        with self.assertRaises(ValueError):
            tracker.step(np.zeros((4, 4), dtype=np.uint8), [])
        with self.assertRaises(ValueError):
            TrackingConfig(gate_fraction=.6)
        image, nodes = scene()
        with self.assertRaises(ValueError):
            tracker.step(image, nodes+[nodes[0]])


if __name__ == '__main__':
    unittest.main()
