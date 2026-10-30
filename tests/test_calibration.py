import contextlib
import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from calibration import CalibrationPanel, length_calibration, load_calibration
from review_session import ReviewSession
from test_temporal_pipeline import make_sequence


class CalibrationTests(unittest.TestCase):
    def test_length_conversion_and_invalid_input(self):
        value = length_calibration([[10, 10], [40, 50]], 5, 'cm', [100, 100])
        self.assertEqual(value['pixel_length'], 50)
        self.assertAlmostEqual(value['units_per_pixel'], .1)
        self.assertAlmostEqual(value['meters_per_pixel'], .001)
        for points, length, unit in (([[1, 1], [1, 1]], 1, 'mm'), ([[1, 1], [5, 5]], -1, 'cm'),
                                     ([[-1, 1], [5, 5]], 1, 'mm'), ([[1, 1], [5, 5]], 1, 'invalid')):
            with self.assertRaises(ValueError):
                length_calibration(points, length, unit, [100, 100])

    def test_draggable_line_dropdown_and_exported_physical_units(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            root = Path(directory)
            results, reference = make_sequence(root)
            session = ReviewSession(results, reference, root, fps=10)
            session.reanalyze()
            fig, ax = plt.subplots()
            image_path = root/'ruler.png'
            Image.new('RGB', (100, 80), 'gray').save(image_path)
            done = []
            figures = plt.get_fignums()
            panel = CalibrationPanel(fig, root, done.append)
            panel.load_image(image_path)
            panel.points = np.array([[10., 10.], [70., 10.]])
            panel.redraw()
            fig.canvas.draw()
            x, y = panel.ax.transData.transform(panel.points[0])
            panel.press(SimpleNamespace(inaxes=panel.ax, button=1, x=x, y=y, xdata=10., ydata=10.))
            panel.move(SimpleNamespace(inaxes=panel.ax, xdata=12., ydata=10.))
            panel.release(None)
            self.assertEqual(panel.points[0, 0], 12.)
            panel.points[0, 0] = 10.
            panel.length.set_val('6')
            panel.units.toggle(None)
            self.assertTrue(panel.units.menu_ax.get_visible())
            panel.units.menu.set_active(list(__import__('calibration').UNITS).index('cm'))
            self.assertEqual(panel.units.value, 'cm')
            panel.apply()
            self.assertEqual(plt.get_fignums(), figures)
            self.assertTrue(ax.get_visible())
            self.assertAlmostEqual(load_calibration(root)['units_per_pixel'], .1)
            self.assertEqual(len(done), 1)
            anchors = session.anchors.copy()
            paths = session.export_csv()
            with paths[0].open() as stream:
                point = next(n for n in csv.DictReader(stream) if n['frame'] == '1' and n['row'] == '0' and n['column'] == '0')
            self.assertAlmostEqual(float(point['x_physical']), 3.6)
            self.assertAlmostEqual(float(point['velocity_physical_x']), 1.)
            self.assertEqual(point['length_unit'], 'cm')
            self.assertEqual(point['velocity_physical_unit'], 'cm/second')
            self.assertEqual(session.anchors, anchors)
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
