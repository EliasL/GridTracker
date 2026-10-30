"""Exercise Matplotlib event dispatch, including hidden and overlapping controls."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent

from review_session import ReviewSession
from test_temporal_pipeline import make_sequence
from tracking_review import TrackingReview


def mouse(fig, name, point, *, axes=None):
    transform = axes.transAxes if axes is not None else fig.transFigure
    x, y = transform.transform(point)
    event = MouseEvent(name, fig.canvas, x, y, button=1)
    fig.canvas.callbacks.process(name, event)


def click(fig, axes):
    mouse(fig, 'button_press_event', (.5, .5), axes=axes)
    mouse(fig, 'button_release_event', (.5, .5), axes=axes)


class MouseEventTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(plt.close, 'all')
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)
        self.style = matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'})
        self.style.__enter__()
        self.addCleanup(self.style.__exit__, None, None, None)
        root = Path(self.temporary.name)
        results, reference = make_sequence(root)
        self.session = ReviewSession(results, reference, root)
        self.session.reanalyze()
        self.viewer = TrackingReview(self.session.report, self.session)
        self.viewer.slider.set_val(1)

    def test_nodes_are_editable_and_confirmable_without_a_mode_switch(self):
        viewer = self.viewer
        self.assertNotIn('Edit nodes', viewer.buttons)
        self.assertTrue(viewer.slider.ax.get_visible())
        self.assertTrue(viewer.legend.axes.get_visible())
        editor = viewer.editor
        before = editor.nodes[0]['x']
        x, y = viewer.ax.transData.transform([before, editor.nodes[0]['y']])
        for name, px in (('button_press_event', x), ('motion_notify_event', x+8), ('button_release_event', x+8)):
            event = MouseEvent(name, viewer.fig.canvas, px, y, button=1)
            viewer.fig.canvas.callbacks.process(name, event)
        self.assertGreater(editor.nodes[0]['x'], before)
        changed = editor.nodes[0]['x']
        self.assertNotIn(1, self.session.anchors)
        click(viewer.fig, viewer.buttons['Confirm frame'].ax)
        self.assertIsNone(viewer.fig.canvas.mouse_grabber)
        self.assertIn(1, self.session.anchors)
        self.assertAlmostEqual(self.session.anchors[1]['nodes'][0]['x'], changed)
        self.assertEqual(viewer.mode, 'review')
        self.assertTrue(viewer.editor.active)
        viewer.slider.set_val(1)
        self.assertTrue(viewer.editor.active)
        click(viewer.fig, viewer.buttons['Confirm frame'].ax)

    def test_hidden_slider_does_not_grab_clicks_in_calibration(self):
        viewer = self.viewer
        click(viewer.fig, viewer.buttons['Calibrate length'].ax)
        original = viewer.slider.val
        mouse(viewer.fig, 'button_press_event', (.5, .185))
        self.assertIsNone(viewer.fig.canvas.mouse_grabber)
        mouse(viewer.fig, 'button_release_event', (.5, .185))
        self.assertEqual(viewer.slider.val, original)
        click(viewer.fig, viewer.calibration_panel.buttons[1].ax)
        self.assertEqual(viewer.mode, 'review')

    def test_hidden_units_menu_ignores_clicks_until_opened(self):
        viewer = self.viewer
        click(viewer.fig, viewer.buttons['Calibrate length'].ax)
        panel = viewer.calibration_panel
        position = panel.units.menu.labels[-1].get_position()
        mouse(viewer.fig, 'button_press_event', position, axes=panel.units.menu_ax)
        mouse(viewer.fig, 'button_release_event', position, axes=panel.units.menu_ax)
        self.assertEqual(panel.units.value, 'mm')
        click(viewer.fig, panel.units.button.ax)
        self.assertTrue(panel.units.menu_ax.get_visible())
        mouse(viewer.fig, 'button_press_event', position, axes=panel.units.menu_ax)
        mouse(viewer.fig, 'button_release_event', position, axes=panel.units.menu_ax)
        self.assertEqual(panel.units.value, 'in')
        self.assertFalse(panel.units.menu_ax.get_visible())
        click(viewer.fig, panel.buttons[1].ax)

    def test_reanalysis_disables_controls_and_restores_them_after_failure(self):
        viewer = self.viewer
        checked = viewer.checks.get_status()
        def processing(progress):
            self.assertFalse(viewer.slider.get_active())
            click(viewer.fig, viewer.buttons['Confirm frame'].ax)
            self.assertEqual(viewer.mode, 'processing')
            self.assertIsNone(viewer.editor)
            raise ValueError('test failure')
        from unittest.mock import patch
        with patch.object(self.session, 'reanalyze', side_effect=processing):
            with self.assertRaisesRegex(ValueError, 'test failure'):
                viewer.reanalyze()
        self.assertEqual(viewer.mode, 'review')
        self.assertTrue(viewer.slider.get_active())
        self.assertEqual(viewer.checks.get_status(), checked)
        self.assertTrue(viewer.editor.active)
        self.assertEqual(viewer.mode, 'review')

    def test_text_focus_does_not_edit_nodes_and_dispose_restores_shortcuts(self):
        viewer = self.viewer
        original = list(matplotlib.rcParams['keymap.back'])
        editor = viewer.editor
        editor.selected = 0
        row = editor.nodes[0]['row']
        click(viewer.fig, editor.fps_box.ax)
        self.assertTrue(editor.fps_box.capturekeystrokes)
        x, y = editor.ax.transAxes.transform((.5, .5))
        viewer.fig.canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', viewer.fig.canvas, '.', x, y))
        self.assertEqual(editor.fps_box.text, '.')
        self.assertEqual(editor.nodes[0]['row'], row)
        viewer.slider.set_val(2)
        self.assertFalse(editor.fps_box.capturekeystrokes)
        self.assertEqual(matplotlib.rcParams['keymap.back'], original)
        self.assertIsNone(viewer.fig.canvas.mouse_grabber)


if __name__ == '__main__':
    unittest.main()
