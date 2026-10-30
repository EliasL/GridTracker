import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

from PIL import Image

from session_data import label_node, validate_nodes
from session_data import load_reference, save_reference, load_ui_state
from reference_editor import ReferenceEditor


class ReferenceTests(unittest.TestCase):
    def result(self, directory):
        image = Path(directory)/'frame.png'
        Image.new('RGB', (100, 100), 'gray').save(image)
        result = dict(source=str(image), image_size=[100, 100], width=4,
                      maxima={'x': dict(clean=[20, 70]), 'y': dict(clean=[20, 70])},
                      final_nodes=[dict(id=1, x=20., y=20., kind='interior')])
        return result, [label_node(n, result) for n in result['final_nodes']]

    def test_approval_is_explicit_and_bound_to_pixels_and_analysis(self):
        with TemporaryDirectory() as directory:
            result, nodes = self.result(directory)
            path = Path(directory)/'reference.json'
            save_reference(path, result, nodes, approved=False)
            with self.assertRaisesRegex(ValueError, 'explicitly approved'):
                load_reference(path, result)
            save_reference(path, result, nodes, approved=True)
            self.assertEqual(load_reference(path, result)['nodes'], nodes)
            result['width'] = 5
            with self.assertRaisesRegex(ValueError, 'different image pixels or detection'):
                load_reference(path, result)
            result['width'] = 4
            Image.new('RGB', (100, 100), 'black').save(result['source'])
            with self.assertRaisesRegex(ValueError, 'different image pixels'):
                load_reference(path, result)

    def test_duplicate_grid_identity_raises(self):
        node = dict(id=1, x=20., y=20., kind='interior', row=0, column=0)
        with self.assertRaisesRegex(ValueError, 'Duplicate grid identity'):
            validate_nodes([node, dict(node, id=2, x=21.)], [100, 100])

    def test_edge_metadata_and_grid_order_are_checked(self):
        node = dict(id=1, x=20., y=20., kind='interior', row=0, column=0)
        edge = dict(id=2, x=20., y=0., kind='edge', side='top', row=None, column=0, scan_axes=['x'])
        validate_nodes([node, edge], [100, 100])
        with self.assertRaisesRegex(ValueError, 'boundary metadata'):
            validate_nodes([node, dict(edge, y=2)], [100, 100])
        with self.assertRaisesRegex(ValueError, 'Crossed identities'):
            validate_nodes([node, dict(node, id=3, column=1, x=10),
                            dict(node, id=4, row=1, y=60),
                            dict(node, id=5, row=1, column=1, x=70, y=60)], [100, 100])

    def test_editing_drag_add_delete_undo_and_unapproved_resume(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with TemporaryDirectory() as directory, matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            result, _ = self.result(directory)
            result['candidate_nodes'] = result['final_nodes']
            editor = ReferenceEditor(result, Path(directory)/'reference.json')
            editor.fig.canvas.draw()
            px, py = editor.ax.transData.transform([20, 20])
            editor.press(SimpleNamespace(inaxes=editor.ax, x=px, y=py, button=1))
            editor.move(SimpleNamespace(inaxes=editor.ax, xdata=23., ydata=21.))
            editor.release(None)
            self.assertEqual(editor.nodes[0]['x'], 23.)
            editor.key(SimpleNamespace(key='a', inaxes=editor.ax, xdata=70., ydata=70.))
            self.assertEqual(len(editor.nodes), 2)
            editor.key(SimpleNamespace(key='delete'))
            self.assertEqual(len(editor.nodes), 1)
            editor.key(SimpleNamespace(key='cmd+z'))
            self.assertEqual(len(editor.nodes), 2)
            editor.close(None)
            self.assertFalse(editor.path.exists())
            resumed = ReferenceEditor(result, editor.path)
            self.assertEqual(resumed.nodes, editor.nodes)
            resumed.approve(None)
            self.assertTrue(load_reference(editor.path, result)['approved'])
            plt.close('all')

    def test_autosave_survives_without_a_close_event_and_preserves_labels(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with TemporaryDirectory() as directory, matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            result, _ = self.result(directory)
            result['candidate_nodes'] = result['final_nodes']
            path = Path(directory)/'reference.json'
            editor = ReferenceEditor(result, path)
            editor.key(SimpleNamespace(key='a', inaxes=editor.ax, xdata=70., ydata=70.))
            editor.key(SimpleNamespace(key=']'))
            self.assertEqual(json.loads(editor.draft_path.read_text())['nodes'][-1]['column'], 2)
            editor.dragging = True
            editor.move(SimpleNamespace(inaxes=editor.ax, xdata=72., ydata=73.))
            editor.flush_autosave()  # The same callback runs every 500 ms in the GUI.
            editor.fps_box.set_val('1e')  # An incomplete input is still saved as a draft.
            saved = json.loads(editor.draft_path.read_text())
            self.assertEqual((saved['nodes'][-1]['x'], saved['nodes'][-1]['y']), (72., 73.))
            self.assertEqual(saved['editor_state']['fps_text'], '1e')
            self.assertFalse(saved['approved'])
            editor.dispose()  # Simulate disappearing without the normal close callback.
            resumed = ReferenceEditor(result, path)
            self.assertEqual(resumed.nodes, saved['nodes'])
            self.assertEqual(resumed.nodes[resumed.selected]['column'], 2)
            self.assertEqual(resumed.fps_box.text, '1e')
            self.assertFalse(path.exists())
            plt.close('all')

    def test_older_sessions_resume_the_latest_unconfirmed_draft(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result, nodes = self.result(directory)
            result['frame'] = 4
            save_reference(root/'reviews'/'frame_04.json', result, nodes, True)
            save_reference(root/'reviews'/'frame_04.draft.json', result, [dict(nodes[0], x=23.)], False)
            self.assertEqual(load_ui_state(root, 10), dict(frame=4, mode='edit'))


if __name__ == '__main__':
    unittest.main()
