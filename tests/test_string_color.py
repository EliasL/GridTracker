import unittest

import numpy as np
from PIL import Image

from string_color import profile_color_grayscale


class ProfileColorTests(unittest.TestCase):
    def test_black_samples_are_counted_and_discarded_but_all_black_raises(self):
        pixels = np.zeros((40, 40, 3), dtype=np.uint8)
        pixels[6:15, 6:15] = (70, 30, 10)
        profiles = [dict(point=[x, 10], normal=[0, 1], width=3, parameters=[0, 0, 1, 0, 1]) for x in (8, 9, 10, 11, 12, 30)]
        gray, model = profile_color_grayscale(Image.fromarray(pixels), profiles)
        self.assertEqual(model['excluded_black_samples'], 3)
        self.assertGreater(gray[10, 10], .9)
        with self.assertRaisesRegex(ValueError, 'nonblack'):
            profile_color_grayscale(Image.fromarray(pixels*0), profiles)

    def test_color_is_learned_and_intensity_is_retained(self):
        for color in ((70, 30, 10), (10, 30, 70)):
            with self.subTest(color=color):
                pixels = np.full((40, 40, 3), 2, dtype=np.uint8)
                pixels[6:15, 6:15] = color
                pixels[6:15, 26:35] = np.array(color) // 2
                pixels[26:35, 6:15] = (30, 70, 10)
                profiles = [{"point": [x, 10], "normal": [0, 1], "width": 3,
                             "parameters": [0, 0, 1, 0, 1]} for x in range(8, 14)]
                gray, model = profile_color_grayscale(Image.fromarray(pixels), profiles)
                self.assertGreater(gray[10, 10], 0.95)
                self.assertAlmostEqual(gray[10, 30], 0.5, delta=0.02)
                self.assertLess(gray[30, 10], 0.05)
                self.assertEqual(model["core_color_samples"], 18)
                self.assertTrue(np.isfinite(gray).all())
                self.assertTrue(((gray >= 0) & (gray <= 1)).all())


if __name__ == "__main__":
    unittest.main()
