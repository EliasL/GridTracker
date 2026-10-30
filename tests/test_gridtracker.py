import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from gridtracker import crossing_evidence, estimate_string_width
from experiments.legacy_growth import grow_grid, extrapolate, fit_side


def synthetic(color=(70, 30, 10), background=(2, 2, 2)):
    image = Image.new("RGB", (320, 240), background)
    draw = ImageDraw.Draw(image)
    for y in range(20, 240, 40):
        draw.line((0, y, 319, y), fill=color, width=3)
    for x in range(20, 320, 40):
        draw.line((x, 0, x, 239), fill=color, width=3)
    return image


class GrowthTests(unittest.TestCase):
    def test_seed_and_growth_until_image_edges(self):
        result = grow_grid(synthetic(), cycles=6)
        self.assertGreaterEqual(result["seed"].shape[0], 3)
        self.assertEqual(result["seed"].shape[0], result["seed"].shape[1])
        self.assertLessEqual(len(result["snapshots"]), 7)
        self.assertEqual(len(result["snapshots"][0]), result["seed"].shape[0] ** 2)
        for old, new in zip(result["snapshots"], result["snapshots"][1:]):
            self.assertTrue(old.keys() <= new.keys())
            for key in old:
                np.testing.assert_array_equal(old[key], new[key])
        final = result["snapshots"][-1]
        self.assertGreater(len(final), 9)
        for point in final.values():
            np.testing.assert_allclose((point - 20) / 40, np.round((point - 20) / 40), atol=0.05)
        for guesses in result["guesses"]:
            for p in guesses.values():
                self.assertTrue(np.all(p >= 0))
                self.assertTrue(np.all(p <= [319, 239]))

    def test_new_nodes_are_one_neighbor_step_from_previous_cycle(self):
        stages = grow_grid(synthetic(), cycles=3)["snapshots"]
        for old, new in zip(stages, stages[1:]):
            added = new.keys() - old.keys()
            self.assertTrue(len({r for r, c in added}) == 1 or len({c for r, c in added}) == 1)
            for r, c in added:
                self.assertTrue(any(key in old for key in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))))

    def test_extrapolation_continues_local_bending(self):
        nodes = {(0, 0): np.array([20., 20.]), (0, 1): np.array([60., 23.])}
        steps = np.array([[0., 40.], [40., 0.]])
        np.testing.assert_array_equal(extrapolate((0, 2), nodes, steps), [100, 26])

    def test_color_neutral_seed(self):
        for color, background in (((10, 20, 80), (2, 2, 2)),
                                  ((70, 70, 70), (2, 2, 2)),
                                  ((10, 10, 10), (90, 90, 90))):
            with self.subTest(color=color):
                result = grow_grid(synthetic(color, background), cycles=1)
                self.assertGreaterEqual(len(result["snapshots"][0]), 9)
                self.assertGreater(len(result["snapshots"][1]), len(result["snapshots"][0]))

    def test_corner_uncertainty_and_threshold(self):
        result = grow_grid(synthetic(), cycles=0)
        records = result["candidates"]
        self.assertEqual({r["corner"] for r in records},
                         {"top-left", "top-right", "bottom-left", "bottom-right"})
        eligible = []
        for corner in {r["corner"] for r in records}:
            failed = False
            accepted = []
            for record in [r for r in records if r["corner"] == corner]:
                self.assertAlmostEqual(record["total"] / record["area"], record["density"])
                failed |= record["density"] > result["max_uncertainty"]
                self.assertEqual(record["accepted"], not failed)
                if record["accepted"]:
                    accepted.append(record)
            if accepted:
                eligible.append(accepted[-1])
        self.assertIn(result["selected"], eligible)
        self.assertAlmostEqual(result["selected"]["density"], min(r["density"] for r in eligible))

    def test_smoothness_between_new_nodes_is_optimized_jointly(self):
        nodes = {(r, 0): np.array([20., 20. + r * 40]) for r in range(4)}
        guesses = {(r, 1): np.array([60., y]) for r, y in enumerate([20, 66, 94, 140])}
        evidence = np.ones((180, 120))
        fitted = fit_side(guesses, nodes, evidence, evidence, np.array([[0., 40.], [40., 0.]]), 1, 0.35)
        initial_y = np.array([guesses[r, 1][1] for r in range(4)])
        final_y = np.array([fitted[r, 1][1] for r in range(4)])
        self.assertLess(np.linalg.norm(np.diff(final_y, n=2)), 0.1)
        self.assertGreater(np.linalg.norm(np.diff(initial_y, n=2)), 10)
        for r in range(4):
            np.testing.assert_array_equal(nodes[r, 0], [20, 20 + r * 40])

    def test_nearest_edge_is_finished_before_turning(self):
        xx, yy = np.meshgrid([180, 220, 260, 300], [20, 60, 100, 140])
        seed = np.stack((xx, yy), axis=-1).astype(float)
        record = {"corner": "top-right", "n": 4}
        with patch("experiments.legacy_growth.find_seed", return_value=(seed, record, [record])):
            result = grow_grid(synthetic())
        self.assertEqual(result["sides"], ["down"] * 2 + ["left"] * 4)
        self.assertEqual(len(result["snapshots"][-1]), 48)

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

    def test_uncertain_growth_is_skipped(self):
        image = synthetic()
        ImageDraw.Draw(image).rectangle((0,0,159,239), fill=(2,2,2))
        xx, yy = np.meshgrid([180,220,260,300], [20,60,100,140])
        seed = np.stack((xx, yy), axis=-1).astype(float)
        record = {"corner": "top-right", "n": 4}
        with patch("experiments.legacy_growth.find_seed", return_value=(seed, record, [record])):
            result = grow_grid(image)
        self.assertTrue(result["rejected"])
        final = result["snapshots"][-1]
        for rejected in result["rejected"]:
            self.assertNotIn((rejected["row"], rejected["column"]), final)
        self.assertTrue(all(score <= result["max_uncertainty"] for score in result["accepted_scores"].values()))
        self.assertLess(len(final), 48)

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            grow_grid(Image.new("RGB", (320, 240)))
        for options in ({"cycles": -1}, {"smoothness": -1}, {"search_fraction": 1}, {"max_uncertainty": 0}):
            with self.assertRaises(ValueError):
                grow_grid(synthetic(), **options)


if __name__ == "__main__":
    unittest.main()
