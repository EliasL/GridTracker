from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import cv2
from PIL import Image

from test_curve_intersections import example_report
from video_export import export_video


class VideoExportTests(unittest.TestCase):
    def report(self, root):
        report = example_report()
        source = root/'source.png'
        Image.new('RGB', (101, 101)).save(source)
        for frame in report['frames']:
            frame['source'] = str(source)
        return report

    def test_mp4_contains_fitted_crossings_and_preserves_missing_data(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.report(root)
            report['frames'][1]['curves'] = report['frames'][1]['curves'][:1]
            original = deepcopy(report)
            calls = []
            path = export_video(report, root/'crossings.mp4', playback_fps=24,
                                progress=lambda done, total: calls.append((done, total)))
            capture = cv2.VideoCapture(str(path))
            try:
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 24)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                for index in range(3):
                    valid, pixels = capture.read()
                    self.assertTrue(valid)
                    self.assertEqual(pixels.shape, (102, 102, 3))
                    x, y = 25+2*index, 40+3*index
                    if index == 1:
                        self.assertLess(pixels[y, x].max(), 30)
                    else:
                        self.assertGreater(pixels[y, x].min(), 190)
                    self.assertLess(pixels[7, 5].max(), 30)  # Guide points are not exported.
                self.assertFalse(capture.read()[0])
            finally:
                capture.release()
            self.assertEqual(report, original)
            self.assertEqual(calls[-1], (3, 3))
            self.assertFalse(list(root.glob('*.csv')))

    def test_failed_export_keeps_existing_video_and_cleans_temporary_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.report(root)
            path = root/'crossings.mp4'
            path.write_bytes(b'existing video')
            bad_source = root/'wrong_size.png'
            Image.new('RGB', (102, 101)).save(bad_source)
            report['frames'][1]['source'] = str(bad_source)
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                export_video(report, path)
            self.assertEqual(path.read_bytes(), b'existing video')
            self.assertEqual(list(root.glob('*.mp4')), [path])
            writer = Mock()
            writer.isOpened.return_value = False
            with patch('video_export.cv2.VideoWriter', return_value=writer):
                with self.assertRaisesRegex(RuntimeError, 'encoder'):
                    export_video(report, path)
            writer.release.assert_called_once()
            self.assertEqual(path.read_bytes(), b'existing video')
            self.assertEqual(list(root.glob('*.mp4')), [path])

    def test_invalid_playback_rate_or_empty_sequence_raises(self):
        for fps in (0, -1, float('nan'), float('inf')):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                export_video(dict(frames=[]), Path('/tmp/unused.mp4'), playback_fps=fps)
        with self.assertRaisesRegex(ValueError, 'No frames'):
            export_video(dict(frames=[]), Path('/tmp/unused.mp4'))
