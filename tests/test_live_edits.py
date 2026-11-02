from copy import deepcopy
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from review_session import ReviewSession
from session_data import save_reference
from test_temporal_pipeline import make_sequence


class LiveEditTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        self.results, self.reference = make_sequence(self.root, count=5, hidden={2: (5,), 3: (5,)})
        self.session = ReviewSession(self.results, self.reference, self.root)
        self.session.reanalyze()

    def node(self, frame, key):
        return next(n for n in self.session.report['frames'][frame]['nodes'] if n['id'] == key)

    def move(self, frame, key, x, y):
        before = deepcopy(self.session.report['frames'][frame]['nodes'])
        after = deepcopy(before)
        node = next(n for n in after if n['id'] == key)
        node.update(x=x, y=y)
        return self.session.apply_edit(frame, before, after)

    def test_edit_immediately_updates_only_following_missing_positions(self):
        observed = deepcopy(self.node(4, 5))
        unrelated = deepcopy(self.node(2, 6))
        self.move(1, 5, 82., 77.)
        self.assertTrue(self.node(1, 5)['human_edited'])
        self.assertEqual(self.node(1, 5)['measurement_source'], 'human')
        for i in (2, 3):
            node = self.node(i, 5)
            self.assertEqual([node['x'], node['y']], [82., 77.])
            self.assertIsNone(node['measured_xy'])
            self.assertFalse(node['human_edited'])
            self.assertIsNone(node['velocity'])
        self.assertEqual(self.node(2, 6), unrelated)
        self.assertEqual(self.node(4, 5), observed)

    def test_next_frames_reanalysis_uses_previous_frame_edit_without_whole_frame_confirmation(self):
        self.move(1, 5, 82., 77.)
        outside = deepcopy(self.session.report['frames'][4])
        self.session.reanalyze(frame_range=(2, 3))
        self.assertEqual([self.node(2, 5)['x'], self.node(2, 5)['y']], [82., 77.])
        self.assertEqual([self.node(3, 5)['x'], self.node(3, 5)['y']], [82., 77.])
        self.assertEqual(self.session.report['frames'][4], outside)
        self.assertEqual(sorted(self.session.anchors), [0])
        self.assertEqual(set(self.session.edits[1]), {5})

    def test_added_human_node_is_carried_and_survives_reanalysis_and_restart(self):
        before = deepcopy(self.session.report['frames'][1]['nodes'])
        added = dict(id=100, kind='interior', row=4, column=4, x=190., y=185., scan_axes=['x', 'y'])
        self.session.apply_edit(1, before, before+[added])
        self.assertEqual(self.node(2, 100)['x'], 190.)
        self.assertIsNone(self.node(2, 100)['measured_xy'])
        self.session.reanalyze(frame_range=(2, 3))
        self.assertEqual(self.node(2, 100)['x'], 190.)
        reopened = ReviewSession(self.results, self.reference, self.root, report=self.session.report)
        self.assertEqual(reopened.edits[1][100]['x'], 190.)

    def test_later_human_edit_stops_quick_propagation_for_that_node(self):
        self.move(3, 5, 90., 80.)
        self.move(1, 5, 82., 77.)
        self.assertEqual(self.node(2, 5)['x'], 82.)
        self.assertEqual(self.node(3, 5)['x'], 90.)
        self.session.reanalyze()
        self.assertEqual(self.node(1, 5)['x'], 82.)
        self.assertEqual(self.node(3, 5)['x'], 90.)

    def test_deletion_does_not_restore_a_missing_guess_in_following_frames(self):
        before = deepcopy(self.session.report['frames'][1]['nodes'])
        self.session.apply_edit(1, before, [n for n in before if n['id'] != 5])
        for i in (1, 2, 3):
            self.assertFalse(any(n['id'] == 5 for n in self.session.report['frames'][i]['nodes']))
        self.assertEqual(self.node(4, 5)['measurement_source'], 'detection')
        self.session.reanalyze()
        for i in (1, 2, 3):
            self.assertFalse(any(n['id'] == 5 for n in self.session.report['frames'][i]['nodes']))

    def test_human_addition_replaces_an_automatic_duplicate_and_reuses_its_identity(self):
        before = deepcopy(self.session.report['frames'][1]['nodes'])
        added = dict(id=500, kind='interior', row=1, column=1, x=82., y=77., scan_axes=['x', 'y'])
        self.session.apply_edit(1, before, before+[added])
        self.assertEqual(set(self.session.edits[1]), {5})
        self.assertEqual(len(self.session.report['frames'][1]['nodes']), len(before))
        self.assertTrue(self.node(1, 5)['human_edited'])
        self.assertEqual(self.node(2, 5)['x'], 82.)

    def test_held_edge_edits_update_future_curve_fits_without_moving_observations(self):
        before = deepcopy(self.session.report['frames'][1]['nodes'])
        edge = dict(id=100, kind='edge', side='top', row=None, column=0, x=35., y=0., scan_axes=['x'])
        self.session.apply_edit(1, before, before+[edge])
        curves = deepcopy(self.session.report['frames'][2]['curves'])
        observed = deepcopy(self.node(2, 0))
        self.move(1, 100, 40., 0.)
        self.assertEqual(self.node(2, 100)['x'], 40.)
        self.assertEqual(self.node(2, 100)['status'], 'held')
        self.assertIsNone(self.node(2, 100)['measured_xy'])
        self.assertEqual(self.node(2, 0), observed)
        self.assertNotEqual(self.session.report['frames'][2]['curves'], curves)

    def test_migration_does_not_turn_stale_held_positions_into_human_edits(self):
        for index, key, x in ((1, 5, 82.), (2, 6, 120.)):
            nodes = deepcopy(self.session.report['frames'][index]['nodes'])
            next(n for n in nodes if n['id'] == key)['x'] = x
            save_reference(self.root/'reviews'/f'frame_{index:02d}.draft.json', self.results[index], nodes, False,
                           editor_state=dict(modified=True))
        reopened = ReviewSession(self.results, self.reference, self.root, report=self.session.report)
        self.assertEqual(set(reopened.edits[1]), {5})
        self.assertEqual(set(reopened.edits[2]), {6})
        held = next(n for n in reopened.report['frames'][2]['nodes'] if n['id'] == 5)
        self.assertEqual(held['x'], 82.)
        self.assertFalse(held['human_edited'])
        self.assertEqual(len(list((self.root/'reviews/history').glob('editing_policy_migration*'))), 2)

