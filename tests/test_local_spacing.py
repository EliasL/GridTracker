import unittest

import numpy as np

from local_peak_spacing import prune_split_peaks, sliding_spacing


class LocalSpacingTests(unittest.TestCase):
    def test_two_peaks_have_no_triple_to_prune(self):
        clean, spacing, removed = prune_split_peaks([20, 80])
        np.testing.assert_array_equal(clean, [20, 80])
        self.assertEqual(spacing, 60)
        self.assertFalse(removed)

    def test_inserted_peak_is_removed(self):
        kept, reference, removed = prune_split_peaks([10, 60, 85, 110, 160, 210])
        np.testing.assert_array_equal(kept, [10, 60, 110, 160, 210])
        self.assertEqual(reference, 50)
        self.assertEqual([r['peak'] for r in removed], [85])

    def test_regular_and_missing_peak_sequences_are_preserved(self):
        for peaks in ([10,60,110,160,210], [10,110,310,410]):
            kept, _, removed = prune_split_peaks(peaks)
            np.testing.assert_array_equal(kept, peaks)
            self.assertFalse(removed)

    def test_observed_two_extra_x_peaks(self):
        original = [49,98,161,202,264,353,458,553,641,733,822,914,1006,1098,1197]
        kept, _, removed = prune_split_peaks(original)
        self.assertEqual(sorted(r['peak'] for r in removed), [98,202])
        self.assertEqual(len(kept), 13)

    def test_multiple_insertions_are_removed_until_idempotent(self):
        peaks = [0,25,50,75,100,200,300,400,500]
        clean, _, _ = prune_split_peaks(peaks)
        np.testing.assert_array_equal(clean, [0,100,200,300,400,500])
        again, _, removed = prune_split_peaks(clean)
        np.testing.assert_array_equal(again, clean)
        self.assertEqual(removed, [])

    def test_undefined_spacing_is_explicit(self):
        signal = np.zeros(200)
        peaks = [20,110,180]
        with self.assertRaises(ValueError):
            sliding_spacing(signal, peaks=peaks)
        _, _, _, spacing, count = sliding_spacing(signal, peaks=peaks, allow_missing=True)
        np.testing.assert_array_equal(np.isnan(spacing), count < 2)


if __name__ == '__main__':
    unittest.main()
