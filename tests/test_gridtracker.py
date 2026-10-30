import unittest

import numpy as np
from PIL import Image, ImageDraw

from detection import crossing_evidence, estimate_string_width, find_seed, ridge_maps


class DetectionTests(unittest.TestCase):
    def test_cross_filter_rejects_single_lines_and_flat_patches(self):
        cross = np.zeros((81, 81))
        cross[39:42, :] = 1
        single = cross.copy()
        cross[:, 39:42] = 1
        self.assertGreater(crossing_evidence(cross, 3)[40,40], 0.8)
        self.assertLess(crossing_evidence(single, 3)[40,40], 0.1)
        self.assertLess(crossing_evidence(np.ones((81,81)), 3)[40,40], 0.1)

    def test_width_estimation_recovers_known_gaussian_fwhm(self):
        yy, xx = np.indices((180,180))
        profile = np.zeros((180,180))
        for center in (30,70,110,150):
            profile = np.maximum(profile, np.exp(-0.5*((yy-center)/(5/2.35482))**2))
            profile = np.maximum(profile, np.exp(-0.5*((xx-center)/(5/2.35482))**2))
        pixels = np.uint8((0.02+0.3*profile)*255)
        image = Image.fromarray(np.repeat(pixels[...,None], 3, axis=2))
        x, y = np.meshgrid([30,70,110,150], [30,70,110,150])
        width, measurements = estimate_string_width(image, np.stack((x,y),axis=-1))
        self.assertAlmostEqual(width, 5, delta=0.2)
        self.assertGreaterEqual(len(measurements), 6)
        self.assertEqual({m["direction"] for m in measurements}, {"horizontal", "vertical"})


    def test_color_neutral_corner_seed_and_uncertainty(self):
        for color, background in (((10, 20, 80), (2, 2, 2)),
                                  ((70, 70, 70), (2, 2, 2)),
                                  ((10, 10, 10), (90, 90, 90))):
            with self.subTest(color=color):
                image = Image.new('RGB', (320, 240), background)
                draw = ImageDraw.Draw(image)
                for y in range(20, 240, 40):
                    draw.line((0, y, 319, y), fill=color, width=3)
                for x in range(20, 320, 40):
                    draw.line((x, 0, x, 239), fill=color, width=3)
                seed, selected, records = find_seed(ridge_maps(image))
                self.assertGreaterEqual(seed.shape[0], 3)
                eligible = []
                for corner in {r['corner'] for r in records}:
                    failed, accepted = False, []
                    for record in [r for r in records if r['corner'] == corner]:
                        self.assertAlmostEqual(record['total']/record['area'], record['density'])
                        failed |= record['density'] > .45
                        self.assertEqual(record['accepted'], not failed)
                        if not failed:
                            accepted.append(record)
                    if accepted:
                        eligible.append(accepted[-1])
                self.assertAlmostEqual(selected['density'], min(r['density'] for r in eligible))
        with self.assertRaises(ValueError):
            ridge_maps(Image.new('RGB', (320, 240)))
