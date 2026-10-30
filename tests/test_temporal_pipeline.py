import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from session_data import file_hash, label_node
from session_data import save_reference
from run_temporal import propagate


def make_sequence(root, count=3, hidden=None):
    y, x = np.mgrid[:200, :200]
    results = []
    for frame in range(count):
        image = np.zeros((200, 200))
        nodes = []
        for row, py in enumerate((35, 75, 115, 155)):
            for column, px in enumerate((35, 75, 115, 155)):
                identity = row*4+column
                px += frame
                if identity in (hidden or {}).get(frame, ()):
                    continue
                nodes.append(dict(id=identity, x=float(px), y=float(py), kind='interior', spatial_accepted=True))
                image += np.exp(-.5*((x-px)/1.5)**2-.5*((y-py)/9)**2)
                image += np.exp(-.5*((y-py)/1.5)**2-.5*((x-px)/9)**2)
        path = root/f'input_{frame}.png'
        Image.fromarray(np.uint8(np.clip(100*image, 0, 255))).convert('RGB').save(path)
        results.append(dict(frame=frame, source=str(path), source_sha256=file_hash(path), image_size=[200, 200],
                            width=4, final_nodes=nodes, candidate_nodes=nodes,
                            maxima={a: dict(clean=[35, 75, 115, 155]) for a in 'xy'},
                            color_model=dict(target_rgb_chromaticity=[1/3]*3, chromaticity_tolerance=.1, reference_intensity=.7)))
    first = results[0]
    reference = save_reference(root/'reference.json', first, [label_node(n, first) for n in first['final_nodes']], True)
    return results, reference


class TemporalPipelineTests(unittest.TestCase):
    def test_approved_sequence_tracks_occlusion_before_fitting(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            results, reference = make_sequence(root, hidden={1: (5,)})
            with contextlib.redirect_stdout(io.StringIO()):
                report = propagate(results, reference, root, fps=100)
            self.assertEqual(len(report['frames']), 3)
            self.assertEqual(len(report['frames'][0]['curves']), 8)
            missing = next(n for n in report['frames'][1]['nodes'] if n['id'] == 5)
            self.assertIsNone(missing['measured_xy'])
            self.assertFalse(any(5 in c['node_ids'] for c in report['frames'][1]['curves']))
            recovered = next(n for n in report['frames'][2]['nodes'] if n['id'] == 5)
            self.assertEqual(recovered['status'], 'recovered')
            self.assertTrue((root/'tracking.json').exists())
            self.assertGreater((root/'tracking_overview.png').stat().st_size, 1000)

    def test_unapproved_reference_cannot_start_propagation(self):
        with self.assertRaisesRegex(ValueError, 'explicitly approved'):
            propagate([dict(frame=0)], dict(approved=False), Path('/tmp'))


if __name__ == '__main__':
    unittest.main()
