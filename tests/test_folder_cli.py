from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from run_temporal import discover_frames


class FolderTests(unittest.TestCase):
    def test_numbered_frames_preserve_gaps_and_sort_naturally(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('frame_10.jpg', 'frame_1.jpg', 'frame_3.jpg'):
                (root/name).write_bytes(b'')
            files, offsets = discover_frames(root, order='filename')
            self.assertEqual([p.name for p in files], ['frame_1.jpg', 'frame_3.jpg', 'frame_10.jpg'])
            self.assertEqual(offsets, [0, 2, 9])

    def test_bad_or_empty_input_raises(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'No image frames'):
                discover_frames(Path(directory))
            with self.assertRaisesRegex(ValueError, 'Not an image folder'):
                discover_frames(Path(directory)/'missing')

    def test_same_folder_outputs_are_not_rediscovered_as_frames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('frame_1.jpg', 'frame_2.jpg', 'tracking_overview.png',
                         'calibration_image.png', 'calibration_input.png'):
                (root/name).touch()
            (root/'frame_00').mkdir()
            (root/'frame_00'/'diagnostic.png').touch()
            files, offsets = discover_frames(root, output=root)
            self.assertEqual([p.name for p in files], ['frame_1.jpg', 'frame_2.jpg'])
            self.assertEqual(offsets, [0, 1])
            # Names are only reserved in the results directory.
            files, _ = discover_frames(root, output=root/'elsewhere')
            self.assertEqual(len(files), 5)


if __name__ == '__main__':
    unittest.main()
