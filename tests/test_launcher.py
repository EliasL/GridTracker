from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from launch_gridtracker import FolderChooser, main


class LauncherTests(unittest.TestCase):
    def chooser(self, source='', output='', same=True):
        chooser = FolderChooser.__new__(FolderChooser)
        chooser.root = Mock()
        chooser.selection = None
        chooser.input = Mock(get=Mock(return_value=source))
        chooser.output = Mock(get=Mock(return_value=output))
        chooser.same_folder = Mock(get=Mock(return_value=same))
        chooser.output_entry = Mock()
        chooser.output_browse = Mock()
        return chooser

    def test_output_follows_input_until_separate_destination_selected(self):
        chooser = self.chooser(source='/chosen/frames')
        chooser.update_output()
        chooser.output.set.assert_called_once_with('/chosen/frames')
        chooser.output_entry.configure.assert_called_with(state='disabled')
        chooser.output.set.reset_mock()
        chooser.same_folder.get.return_value = False
        chooser.update_output()
        chooser.output.set.assert_not_called()
        chooser.output_browse.configure.assert_called_with(state='normal')

    def test_start_validates_input_and_accepts_new_output_folder(self):
        with TemporaryDirectory() as directory:
            source = Path(directory)
            destination = source/'new results'
            chooser = self.chooser(str(source), str(destination), same=False)
            with patch('tkinter.messagebox.showerror') as error:
                chooser.start()
                error.assert_called_once()  # Empty input is not silently accepted.
                chooser.root.destroy.assert_not_called()
            (source/'frame_1.jpg').touch()
            chooser.start()
            self.assertEqual(chooser.selection, (source.resolve(), destination.resolve()))
            chooser.root.destroy.assert_called_once()
            self.assertFalse(destination.exists())  # Run creates it after selection.

    def test_cancel_does_not_start_analysis_and_selection_is_forwarded(self):
        with patch('gui_support.configure_tk'), patch('tkinter.Tk'), \
             patch('launch_gridtracker.FolderChooser') as dialog, patch('run_temporal.main') as run:
            dialog.return_value.selection = None
            main()
            run.assert_not_called()
            dialog.return_value.selection = (Path('/frames'), Path('/results'))
            main()
            run.assert_called_once_with(['/frames', '--output', '/results'])


if __name__ == '__main__':
    unittest.main()
