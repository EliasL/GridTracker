from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from rotation import estimate_rotation, correct_image, crop_size, prepare_frames, RotationEditor
from session_data import file_hash


class RotationTests(unittest.TestCase):
    def grid(self):
        image = Image.new('RGB', (480, 320))
        draw = ImageDraw.Draw(image)
        for x in range(20, 480, 40):
            draw.line((x, 0, x, 319), fill='white', width=3)
        for y in range(20, 320, 40):
            draw.line((0, y, 479, y), fill='white', width=3)
        return image

    def test_rotation_is_estimated_with_the_correct_sign(self):
        for angle in (-12, -3, 2, 11):
            with self.subTest(angle=angle):
                source = self.grid().rotate(angle, resample=Image.Resampling.BICUBIC)
                self.assertAlmostEqual(estimate_rotation(source), -angle, delta=.1)

    def test_crops_have_no_synthetic_border_and_zero_preserves_every_pixel(self):
        for size in ((480, 320), (100, 300), (201, 199)):
            image = Image.new('RGB', size, 'white')
            for angle in (.5, -3, 15, 40, 45):
                with self.subTest(size=size, angle=angle):
                    corrected = correct_image(image, angle)
                    self.assertEqual(corrected.size, crop_size(size, angle))
                    self.assertEqual(int(np.asarray(corrected).min()), 255)
        image = self.grid()
        np.testing.assert_array_equal(correct_image(image, 0), image)

    def test_preparation_is_saved_reused_and_keeps_originals_unchanged(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [root/f'frame_{i}.png' for i in range(2)]
            for path in sources:
                self.grid().save(path)
            hashes = [file_hash(p) for p in sources]
            output = root/'results'
            output.mkdir()
            first = prepare_frames(sources, output, angle=-3, interactive=False)
            times = [p.stat().st_mtime_ns for p in first]
            self.assertEqual(first, prepare_frames(sources, output, interactive=False))
            self.assertEqual(times, [p.stat().st_mtime_ns for p in first])
            self.assertEqual(hashes, [file_hash(p) for p in sources])
            with self.assertRaisesRegex(ValueError, 'new results folder'):
                prepare_frames(sources, output, angle=0, interactive=False)

    def test_threshold_requires_review_and_existing_references_keep_their_coordinates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.png'
            self.grid().save(source)
            output = root/'results'
            output.mkdir()
            with patch('rotation.estimate_rotation', return_value=2.):
                with self.assertRaisesRegex(ValueError, 'review the crop'):
                    prepare_frames([source], output, interactive=False)
            reference = output/'reference.json'
            reference.write_text('{}')
            with patch('rotation.estimate_rotation', side_effect=AssertionError('Do not rotate saved coordinates')):
                self.assertEqual(prepare_frames([source], output, reference=reference), [source])

    def test_editor_uses_the_typed_angle_and_shows_the_exact_crop_dimensions(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        editor = RotationEditor(self.grid(), 3.)
        editor.angle.set_val('-2')
        editor.apply()
        self.assertEqual(editor.accepted, -2.)
        self.assertEqual(editor.preview.get_array().shape[:2], tuple(reversed(crop_size((480, 320), -2))))
        plt.close('all')

    def test_interactive_setup_prefills_the_estimate_and_uses_the_user_correction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.png'
            self.grid().save(source)
            output = root/'results'
            output.mkdir()
            with patch('rotation.estimate_rotation', return_value=3.), patch('rotation.RotationEditor') as editor:
                editor.return_value.show.return_value = -2.
                paths = prepare_frames([source], output, backend='Agg')
                self.assertEqual(editor.call_args.args[1], 3.)
            self.assertEqual(json.loads((output/'rotation.json').read_text())['angle'], -2.)
            with Image.open(paths[0]) as image:
                self.assertEqual(image.size, crop_size((480, 320), -2.))
            canceled = root/'canceled'
            canceled.mkdir()
            with patch('rotation.estimate_rotation', return_value=3.), patch('rotation.RotationEditor') as editor:
                editor.return_value.show.return_value = None
                with self.assertRaisesRegex(SystemExit, 'canceled'):
                    prepare_frames([source], canceled, backend='Agg')
            self.assertFalse((canceled/'rotation.json').exists())
