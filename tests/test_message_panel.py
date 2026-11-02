from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from message_panel import MessagePanel, open_folder


class MessageTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_native_status_wraps_supports_copy_and_links_to_the_folder(self):
        fig = plt.figure(figsize=(16, 9))
        with patch.object(fig.canvas, 'get_tk_widget', create=True, return_value=Mock()), \
             patch('tkinter.Text') as text, patch('message_panel.open_folder') as open_path:
            panel = MessagePanel(fig, [.025, .005, .75, .075])
            panel.set('Exported to /a/long folder/with spaces', path=Path('/a/long folder/with spaces'))
            self.assertEqual(text.call_args.kwargs['wrap'], 'word')
            text.return_value.tag_add.assert_called_with('folder', '1.0+12c', '1.0+38c')
            text.return_value.get.return_value = 'selected message'
            self.assertEqual(panel.copy(), 'break')
            text.return_value.clipboard_append.assert_called_once_with('selected message')
            callback = next(call.args[2] for call in text.return_value.tag_bind.call_args_list
                            if call.args[:2] == ('folder', '<ButtonRelease-1>'))
            text.return_value.tag_ranges.return_value = ('1.0', '1.5')
            callback(Mock())
            open_path.assert_not_called()
            text.return_value.tag_ranges.return_value = ()
            callback(Mock())
            open_path.assert_called_once_with(Path('/a/long folder/with spaces'))
            placement = text.return_value.place.call_args.kwargs
            self.assertLess(placement['relx']+placement['relwidth'], .8)
            self.assertEqual(text.return_value.configure.call_args.kwargs['state'], 'disabled')
            panel.close()
            text.return_value.destroy.assert_called_once()

    def test_folder_opening_uses_platform_file_manager(self):
        with TemporaryDirectory() as directory:
            path = str(Path(directory).resolve())
            with patch('message_panel.sys.platform', 'darwin'), patch('message_panel.subprocess.Popen') as run:
                open_folder(directory)
                run.assert_called_once_with(['open', path])
            with patch('message_panel.sys.platform', 'win32'), patch('message_panel.os.startfile', create=True) as run:
                open_folder(directory)
                run.assert_called_once_with(path)
