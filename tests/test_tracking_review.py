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

    def test_crossing_toggle_matches_export_and_updates_with_drafts(self):
        import numpy as np
        from matplotlib.collections import PathCollection
        from review_session import curve_crossing_snapshot
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            viewer = TrackingReview(session.report, session)
            viewer.slider.set_val(1)
            index = list(viewer.OPTIONS).index('Grid crossings')
            viewer.checks.set_active(index)
            for edited in (False, True):
                if edited:
                    for node in viewer.editor.nodes:
                        node['x'] += 2
                    viewer.editor.redraw()
                    viewer.editor.autosave()
                snapshot = curve_crossing_snapshot(session.export_snapshot())
                expected = [[n['x'], n['y']] for n in snapshot['frames'][1]['nodes'] if n['status'] == 'curve_crossing']
                crossing = [a for a in viewer.curve_artists if isinstance(a, PathCollection)]
                self.assertEqual(len(crossing), 1)
                np.testing.assert_allclose(crossing[0].get_offsets(), expected)
            viewer.checks.set_active(list(viewer.OPTIONS).index('Guide nodes'))
            self.assertFalse(viewer.editor.points.get_visible())
            before = len(viewer.editor.nodes)
            viewer.editor.selected = 0
            viewer.editor.key(SimpleNamespace(key='delete'))
            self.assertEqual(len(viewer.editor.nodes), before)
            viewer.checks.set_active(index)
            self.assertFalse(any(isinstance(a, PathCollection) for a in viewer.curve_artists))
            self.assertNotIn(1, session.anchors)
            names = [item.get_text() for item in viewer.legend.get_texts()]
            self.assertEqual(len(names), 6)
            self.assertIn('Tracked / confirmed', names)
            viewer.fig.canvas.draw()
            renderer = viewer.fig.canvas.get_renderer()
            bounds = viewer.legend.get_window_extent(renderer)
            panel = viewer.legend.axes.get_window_extent(renderer)
            self.assertGreaterEqual(bounds.x0, panel.x0)
            self.assertLessEqual(bounds.x1, panel.x1)
            self.assertGreaterEqual(bounds.y0, panel.y0)
            self.assertLessEqual(bounds.y1, panel.y1)
            panels = [viewer.checks.ax]+[b.ax for b in viewer.buttons.values()]
            for i, first in enumerate(panels):
                for second in panels[i+1:]:
                    self.assertFalse(first.get_position().overlaps(second.get_position()))
            plt.close('all')

    def test_mp4_button_includes_saved_drafts_without_confirming_or_exporting_csv(self):
        import cv2
        from copy import deepcopy
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            viewer = TrackingReview(session.report, session)
            viewer.slider.set_val(1)
            for node in viewer.editor.nodes:
                node['x'] += 12
            viewer.editor.redraw()
            viewer.editor.commit_edits()
            anchors = deepcopy(session.anchors)
            report = deepcopy(session.report)
            with patch.object(session, 'export_csv', side_effect=AssertionError('CSV must remain separate')):
                viewer.export_video()
            self.assertEqual(session.anchors, anchors)
            self.assertEqual(session.report, report)
            self.assertFalse(list(root.glob('*.csv')))
            self.assertEqual(viewer.mode, 'review')
            self.assertTrue(viewer.editor.active)
            self.assertTrue(viewer.buttons['Export MP4'].get_active())
            capture = cv2.VideoCapture(str(root/'grid_crossings.mp4'))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
                valid, pixels = capture.read()
                self.assertTrue(valid)
                self.assertGreater(pixels[35, 48].min(), 190)
            finally:
                capture.release()
            plt.close('all')

    def test_first_frame_mp4_export_uses_completed_detections_without_approval(self):
        import cv2
        from review_session import export_detection_video
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, _ = make_sequence(root)
            editor = ReferenceEditor(results[0], root/'unapproved.json',
                                     on_export_video=lambda nodes, fps: export_detection_video(results, root, nodes, fps))
            for node in editor.nodes:
                node['x'] += 12
            editor.redraw()
            editor.export_video()
            self.assertIsNone(editor.approved)
            self.assertTrue(editor.path.exists())
            self.assertFalse(list(root.glob('*.csv')))
            capture = cv2.VideoCapture(str(root/'grid_crossings.mp4'))
            try:
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 30)
                valid, pixels = capture.read()
                self.assertTrue(valid)
                self.assertGreater(pixels[35, 47].min(), 190)
            finally:
                capture.release()
            plt.close('all')

    def test_failed_mp4_export_restores_controls_and_editor(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            viewer = TrackingReview(session.report, session)
            with patch.object(session, 'export_video', side_effect=RuntimeError('encoder failed')):
                with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
                    viewer.export_video()
            self.assertEqual(viewer.mode, 'review')
            self.assertTrue(viewer.editor.active)
            self.assertTrue(all(button.get_active() for name, button in viewer.buttons.items() if name != 'Reset edits'))
            self.assertIn('failed', viewer.message.get_text())
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
            self.assertEqual(reference['verification_scope'], 'edited_nodes')
            self.assertFalse(any(n['human_edited'] for n in reference['nodes']))
            self.assertTrue(plt.fignum_exists(figure.number))
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            viewer = TrackingReview(session.report, session, figure=figure)
            self.assertIs(viewer.fig, figure)
            self.assertEqual(plt.get_fignums(), figures)
            plt.close('all')

    def test_colored_legend_and_export_with_automatic_node_confirmation(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            viewer = TrackingReview(session.report, session)
            colors = [to_rgba(t.get_color()) for t in viewer.legend.get_texts()]
            self.assertIn(to_rgba('lime'), colors)
            self.assertIn(to_rgba('yellow'), colors)
            self.assertNotIn('Confirm frame', viewer.buttons)
            viewer.slider.set_val(1)
            viewer.editor.nodes[0]['x'] += 2
            viewer.editor.redraw()
            viewer.editor.release(None)
            self.assertEqual(set(session.edits[1]), {0})
            self.assertEqual(viewer.state['frame'], 1)
            viewer.export()
            self.assertTrue((root/'nodes.csv').exists())
            plt.close('all')

    def test_reset_edits_button_reanalyzes_the_selected_frame(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root)
            session.reanalyze()
            viewer = TrackingReview(session.report, session)
            self.assertFalse(viewer.buttons['Reset edits'].get_active())
            viewer.slider.set_val(1)
            viewer.editor.nodes[0]['x'] += 2
            viewer.editor.redraw()
            viewer.editor.release(None)
            self.assertTrue(viewer.buttons['Reset edits'].get_active())
            viewer.unmark()
            self.assertNotIn(1, session.edits)
            self.assertFalse(session.dirty)
            self.assertFalse(viewer.buttons['Reset edits'].get_active())
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
            viewer.editor.commit_edits()
            self.assertEqual(viewer.mode, 'review')
            self.assertIsNotNone(viewer.editor)
            self.assertTrue(viewer.editor.active)
            self.assertIn(1, session.edits)
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
