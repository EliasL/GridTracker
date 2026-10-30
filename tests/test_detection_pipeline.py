"""Exercise the detector on a real frame and verify diagnostics are optional."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pipeline import run_frame

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'testData'/'WhatsApp Image 2026-10-28 at 12.13.25.jpeg'


class DetectionPipelineTests(unittest.TestCase):
    def test_detector_matches_baseline_without_constructing_diagnostic_figures(self):
        expected = json.loads((ROOT/'tests/fixtures/detection.json').read_text())
        with TemporaryDirectory() as directory:
            with patch('matplotlib.pyplot.subplots', side_effect=AssertionError('Diagnostics disabled')):
                result = run_frame((0, SOURCE, directory), diagnostics=False)
            self.assertEqual(result['artifacts'], [])
            self.assertEqual(list((Path(directory)/'frame_00').iterdir()), [Path(directory)/'frame_00/result.json'])
        for key in ('image_size', 'maxima', 'final_nodes'):
            self.assertEqual(result[key], expected[key])
        for key in ('width', 'threshold'):
            self.assertAlmostEqual(result[key], expected[key], places=10)
        self.assertEqual(len(result['curves']), len(expected['curves']))
        for actual, previous in zip(result['curves'], expected['curves']):
            for key in ('family', 'peak', 'node_ids'):
                self.assertEqual(actual[key], previous[key])
            np.testing.assert_allclose(actual['control_points'], previous['control_points'], rtol=0, atol=1e-5)
            self.assertAlmostEqual(actual['rmse'], previous['rmse'], places=7)

    def test_diagnostics_preserve_detections_and_create_valid_artifacts(self):
        expected = json.loads((ROOT/'tests/fixtures/detection.json').read_text())
        with TemporaryDirectory() as directory:
            result = run_frame((0, SOURCE, directory), diagnostics=True, fit_curves=False)
            self.assertEqual(result['final_nodes'], expected['final_nodes'])
            self.assertEqual(result['maxima'], expected['maxima'])
            self.assertEqual(result['curves'], [])
            self.assertEqual(len(result['artifacts']), 13)
            for artifact in result['artifacts']:
                with Image.open(artifact['path']) as image:
                    self.assertGreater(min(image.size), 100)
                    image.verify()
            for name in ('scalar.npy', 'crossing_scores.npy'):
                scores = np.load(Path(directory)/'frame_00'/name)
                self.assertEqual(scores.shape, tuple(reversed(result['image_size'])))
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(((0 <= scores) & (scores <= 1)).all())
