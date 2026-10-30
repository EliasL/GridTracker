import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from reference_editor import ReferenceEditor
from review_session import ReviewSession
from test_temporal_pipeline import make_sequence
from tracking_review import TrackingReview


class ReviewWidgetTests(unittest.TestCase):
    def test_drafts_survive_frame_switching_and_export_together(self):
        import csv
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            viewer = TrackingReview(session.report, session)
            for index in (1, 2):
                viewer.slider.set_val(index)
                for node in viewer.editor.nodes:
                    node['x'] += 2
                viewer.editor.redraw()
            viewer.slider.set_val(1)
            self.assertAlmostEqual(viewer.editor.nodes[0]['x'], 38.)
            viewer.export()
            with (root/'nodes.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for index in (1, 2):
                row = next(r for r in rows if r['frame'] == str(index) and r['row'] == r['column'] == '0')
                self.assertAlmostEqual(float(row['x']), 37.+index)
                self.assertEqual(row['human_verified'], 'False')
            self.assertEqual(sorted(session.anchors), [0])
            self.assertTrue(all(label.get_color() == '#222222' for label in viewer.checks.labels))
            viewer.toggle_help()
            self.assertTrue(viewer.help.visible)
            self.assertFalse(viewer.editor.active)
            viewer.toggle_help()
            self.assertTrue(viewer.editor.active)
            plt.close('all')

    def test_next_ten_button_requests_frames_after_current_selection(self):
        from unittest.mock import patch
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            viewer = TrackingReview(session.report, session)
            viewer.slider.set_val(1)
            with patch.object(viewer, 'reanalyze') as run:
                viewer.reanalyze_next()
                run.assert_called_once_with(frame_range=(2, 2))
            viewer.slider.set_val(2)
            with patch.object(viewer, 'reanalyze') as run:
                viewer.reanalyze_next()
                run.assert_not_called()
            plt.close('all')

    def test_first_frame_setup_transitions_to_review_in_the_same_figure(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, _ = make_sequence(root)
            editor = ReferenceEditor(results[0], root/'new_reference.json', keep_open=True)
            figure = editor.fig
            figures = plt.get_fignums()
            with patch.object(figure.canvas, 'start_event_loop', side_effect=lambda _: editor.approve(None)):
                reference = editor.show()
            self.assertTrue(reference['approved'])
            self.assertTrue(plt.fignum_exists(figure.number))
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            viewer = TrackingReview(session.report, session, figure=figure)
            self.assertIs(viewer.fig, figure)
            self.assertEqual(plt.get_fignums(), figures)
            plt.close('all')

    def test_colored_legend_and_export_without_confirmation(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            viewer = TrackingReview(session.report, session)
            colors = [to_rgba(t.get_color()) for t in viewer.legend.get_texts()]
            self.assertIn(to_rgba('lime'), colors)
            self.assertIn(to_rgba('yellow'), colors)
            self.assertIn(to_rgba('red'), colors)
            self.assertTrue(viewer.buttons['Export CSV'].get_active())
            viewer.slider.set_val(1)
            viewer.confirm()
            self.assertEqual(sorted(session.anchors), [0, 1])
            self.assertEqual(viewer.state['frame'], 2)
            viewer.confirm()
            self.assertTrue(viewer.buttons['Export CSV'].get_active())
            viewer.export()
            self.assertTrue((root/'nodes.csv').exists())
            plt.close('all')

    def test_unmark_button_reanalyzes_the_selected_frame(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            viewer = TrackingReview(session.report, session)
            self.assertFalse(viewer.buttons['Unmark frame'].get_active())
            viewer.slider.set_val(1)
            viewer.confirm()
            viewer.slider.set_val(1)
            self.assertTrue(viewer.buttons['Unmark frame'].get_active())
            viewer.unmark()
            self.assertNotIn(1, session.anchors)
            self.assertFalse(session.report['frames'][1]['human_verified'])
            self.assertFalse(session.dirty)
            self.assertFalse(viewer.buttons['Unmark frame'].get_active())
            plt.close('all')

    def test_always_editable_export_and_calibration_reuse_one_figure(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            viewer = TrackingReview(session.report, session)
            number = viewer.fig.number
            figures = plt.get_fignums()
            viewer.slider.set_val(1)
            self.assertEqual(viewer.editor.fig.number, number)
            self.assertEqual(plt.get_fignums(), figures)
            editor = viewer.editor
            editor.nodes[0]['x'] += 2
            editor.redraw()
            viewer.export()
            self.assertTrue((root/'nodes.csv').exists())
            self.assertEqual(sorted(session.anchors), [0])
            viewer.calibrate()
            self.assertEqual(plt.get_fignums(), figures)
            viewer.calibration_panel.cancel()
            viewer.confirm()
            self.assertEqual(viewer.mode, 'review')
            self.assertIsNotNone(viewer.editor)
            self.assertTrue(viewer.editor.active)
            self.assertIn(1, session.anchors)
            self.assertEqual(plt.get_fignums(), figures)
            viewer.calibrate()
            viewer.calibration_panel.cancel()
            self.assertEqual(viewer.mode, 'review')
            plt.close('all')

    def test_reopening_restores_autosaved_edits_without_an_edit_mode(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            viewer = TrackingReview(session.report, session)
            viewer.slider.set_val(2)
            editor = viewer.editor
            editor.selected, editor.dragging = 0, True
            editor.move(SimpleNamespace(inaxes=editor.ax, xdata=39., ydata=35.))
            editor.release(None)
            editor.dispose()
            plt.close('all')
            reopened = ReviewSession(results, reference, root, report=session.report)
            restored = TrackingReview(reopened.report, reopened)
            self.assertEqual(restored.state['frame'], 2)
            self.assertEqual(restored.mode, 'review')
            self.assertEqual(restored.editor.nodes[0]['x'], 39.)
            self.assertEqual(sorted(reopened.anchors), [0])
            self.assertFalse(restored.editor.approved)
            plt.close('all')

    def test_frame_rate_box_is_optional_and_does_not_trigger_node_edit_keys(self):
        with TemporaryDirectory() as directory, matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, _ = make_sequence(root)
            editor = ReferenceEditor(results[0], root/'new_reference.json')
            editor.selected = 0
            previous = editor.nodes[0]['row']
            editor.key(SimpleNamespace(key='.', inaxes=editor.fps_box.ax))
            self.assertEqual(editor.nodes[0]['row'], previous)
            editor.fps_box.set_val('2000')
            editor.approve(None)
            self.assertEqual(editor.approved['fps'], 2000.)
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
