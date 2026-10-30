from copy import deepcopy
import contextlib
import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from session_data import label_node
from session_data import save_reference
from review_session import ReviewSession, update_kinematics
from test_temporal_pipeline import make_sequence


class ReviewSessionTests(unittest.TestCase):
    def test_partial_reanalysis_changes_only_requested_unconfirmed_frames(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            results, reference = make_sequence(root, count=14)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            session.confirm(3, self.approve(root, results[3]))
            session.confirm(8, self.approve(root, results[8]))
            before = deepcopy(session.report['frames'])
            calls = []
            session.reanalyze(frame_range=(4, 13), progress=lambda done, total: calls.append((done, total)))
            for index in (0, 1, 2, 3, 8):
                self.assertEqual(session.report['frames'][index], before[index])
            self.assertEqual(session.report['reanalyzed_range'], [4, 13])
            self.assertEqual(calls[-1], (11, 11))
            self.assertTrue(session.dirty)  # A partial run cannot clear a global correction.

    def approve(self, root, result, fps=None, omit=()):
        nodes = [label_node(n, result) for n in result['final_nodes'] if n['id'] not in omit]
        return save_reference(root/f"approved_{result['frame']}.json", result, nodes, True, fps)

    def test_new_human_anchor_improves_earlier_frames_without_changing_confirmed_frames(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            results, _ = make_sequence(root, count=4)
            first = self.approve(root, results[0], omit=(5,))
            session = ReviewSession(results, first, root)
            initial = session.reanalyze()
            self.assertFalse(any(n['id'] == 5 for n in initial['frames'][1]['nodes']))
            last = self.approve(root, results[3])
            session.confirm(3, last)
            frozen = deepcopy(session.anchors)
            report = session.reanalyze()
            restored = next(n for n in report['frames'][1]['nodes'] if n['id'] == 5)
            self.assertIsNotNone(restored['measured_xy'])
            self.assertEqual(restored['temporal_direction'], 'backward')
            self.assertEqual(session.anchors, frozen)
            self.assertFalse(any(n['id'] == 5 for n in report['frames'][0]['nodes']))
            for index in (0, 3):
                expected = [(n['id'], n['x'], n['y']) for n in session.anchors[index]['nodes']]
                self.assertEqual([(n['id'], n['x'], n['y']) for n in report['frames'][index]['nodes']], expected)
            reopened = ReviewSession(results, first, root, report=report)
            self.assertEqual(reopened.anchors, frozen)

    def test_confirmed_absence_does_not_delete_a_track_from_future_frames(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            results, reference = make_sequence(root, count=4)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            session.confirm(2, self.approve(root, results[2], omit=(5,)))
            report = session.reanalyze()
            self.assertFalse(any(n['id'] == 5 for n in report['frames'][2]['nodes']))
            restored = next(n for n in report['frames'][3]['nodes'] if n['id'] == 5)
            self.assertEqual(restored['status'], 'recovered')
            for index in (1, 3):
                session.confirm(index, self.approve(root, results[index]))
            paths = session.export_csv()
            with paths[0].open() as stream:
                absent = next(n for n in csv.DictReader(stream) if n['frame'] == '2' and n['row'] == '1' and n['column'] == '1')
            self.assertEqual(absent['status'], 'missing_curve')
            self.assertEqual(absent['x'], '')
            self.assertEqual(absent['observed'], 'False')
            self.assertEqual(absent['position_verified'], 'False')

    def test_export_is_available_before_confirmation_and_uses_confirmed_positions_and_sampling_rate(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            session.export_csv()
            self.assertEqual(sorted(session.anchors), [0])
            for result in results[1:]:
                session.confirm(result['frame'], self.approve(root, result, fps=10))
            paths = session.export_csv()
            with paths[0].open() as stream:
                table = list(csv.DictReader(stream))
            self.assertEqual(len(table), 48)
            node = next(n for n in table if n['frame'] == '1' and n['row'] == '1' and n['column'] == '1')
            self.assertAlmostEqual(float(node['x']), 76.)
            self.assertAlmostEqual(float(node['velocity_x']), 10.)
            self.assertAlmostEqual(float(node['acceleration_x']), 0.)
            self.assertEqual(node['velocity_unit'], 'pixels/second')
            self.assertEqual(node['human_verified'], 'True')
            self.assertTrue(all(p.exists() for p in paths))

    def test_unmark_releases_one_frame_and_preserves_other_anchors_and_old_edits(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            nodes = [label_node(n, results[1]) for n in results[1]['final_nodes']]
            nodes[0]['x'] += 2
            document = save_reference(root/'reviews'/'frame_01.json', results[1], nodes, True)
            save_reference(root/'reviews'/'frame_01.draft.json', results[1], nodes, False)
            session.confirm(1, document)
            session.confirm(2, self.approve(root, results[2]))
            frozen = deepcopy({i: session.anchors[i] for i in (0, 2)})
            self.assertTrue(session.unmark(1))
            self.assertFalse(session.report['frames'][1]['human_verified'])
            self.assertFalse((root/'reviews'/'frame_01.draft.json').exists())
            self.assertTrue(list((root/'reviews'/'history').glob('*')))
            report = session.reanalyze()
            self.assertAlmostEqual(report['frames'][1]['nodes'][0]['x'], 36.)
            self.assertEqual(session.anchors, frozen)
            reopened = ReviewSession(results, reference, root, report=report)
            self.assertEqual(sorted(reopened.anchors), [0, 2])
            with self.assertRaisesRegex(ValueError, 'first frame'):
                session.unmark(0)

    def test_irregular_source_frame_spacing_is_used_for_derivatives(self):
        offsets = [0., 1., 3., 4.]
        report = dict(fps=2., frames=[dict(frame=i, frame_offset=t, nodes=[dict(id=1, kind='interior', measured_xy=[t*t, 0.])]) for i, t in enumerate(offsets)])
        update_kinematics(report)
        np.testing.assert_allclose([f['time'] for f in report['frames']], np.array(offsets)/2)
        np.testing.assert_allclose([f['nodes'][0]['acceleration'][0] for f in report['frames']], 8.)

    def test_inconsistent_persistent_id_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            nodes = [label_node(n, results[1]) for n in results[1]['final_nodes']]
            nodes[0]['id'] = 100
            with self.assertRaisesRegex(ValueError, 'different IDs'):
                session.validate_edit(1, nodes)

    def test_derivatives_do_not_cross_missing_data_or_apply_to_boundary_intersections(self):
        frames = []
        for frame in range(5):
            frames.append(dict(frame=frame, nodes=[
                dict(id=1, kind='interior', measured_xy=[frame**2, 0.] if frame != 2 else None),
                dict(id=2, kind='edge', measured_xy=[frame, 0.])]))
        report = dict(fps=2., frames=frames)
        update_kinematics(report)
        self.assertIsNone(frames[2]['nodes'][0]['velocity'])
        self.assertTrue(all(f['nodes'][0]['acceleration'] is None for f in frames))
        self.assertTrue(all(f['nodes'][1]['velocity'] is None for f in frames))
        full = dict(fps=2., frames=[dict(frame=i, nodes=[dict(id=1, kind='interior', measured_xy=[i*i, 0.])]) for i in range(5)])
        update_kinematics(full)
        np.testing.assert_allclose([f['nodes'][0]['acceleration'][0] for f in full['frames']], 8.)


if __name__ == '__main__':
    unittest.main()
