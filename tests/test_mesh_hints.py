import unittest

from mesh_hints import label_from_mesh, missing_crossings
from reference_editor import validate_nodes


def mesh():
    nodes = []
    for row, y in enumerate((20., 50., 80.)):
        for col, x in enumerate((20., 50., 80.)):
            nodes.append(dict(id=len(nodes), kind='interior', x=x, y=y, row=row, column=col))
        for side, x in (('left', 0.), ('right', 100.)):
            nodes.append(dict(id=len(nodes), kind='edge', side=side, x=x, y=y, row=row, column=None, scan_axes=['y']))
    for col, x in enumerate((20., 50., 80.)):
        for side, y in (('top', 0.), ('bottom', 100.)):
            nodes.append(dict(id=len(nodes), kind='edge', side=side, x=x, y=y, row=None, column=col, scan_axes=['x']))
    return nodes


class MeshHintTests(unittest.TestCase):
    def test_missing_crossing_is_localized_from_boundary_grid(self):
        nodes = [n for n in mesh() if not (n['row'] == n['column'] == 1)]
        self.assertEqual(missing_crossings(nodes, [101, 101], 2), [dict(row=1, column=1, x=50., y=50.)])
        self.assertFalse(missing_crossings(mesh(), [101, 101], 2))

    def test_boundary_changes_update_dimensions_and_new_string_identity(self):
        nodes = mesh()
        result = dict(width=2, image_size=[101, 101], maxima={a: dict(clean=[20, 50, 80]) for a in 'xy'})
        edge = label_from_mesh(dict(id=100, kind='edge', side='top', x=65., y=0., scan_axes=['x']), nodes, result)
        self.assertEqual(edge['column'], 3)
        nodes.append(edge)
        self.assertEqual(len(missing_crossings(nodes, [101, 101], 2)), 3)
        validate_nodes(nodes, [101, 101])
        nodes.remove(edge)
        self.assertFalse(missing_crossings(nodes, [101, 101], 2))

    def test_clipped_boundary_cells_and_extra_partial_nodes_are_tolerated(self):
        nodes = mesh()
        for node in nodes:
            if node['column'] == 0:
                node['x'] = 2.
        nodes = [n for n in nodes if not (n['kind'] == 'interior' and n['column'] == 0)]
        nodes.append(dict(id=100, kind='interior', x=99., y=97., row=20, column=20))
        self.assertFalse(missing_crossings(nodes, [101, 101], 2))
        validate_nodes(nodes, [101, 101])

    def test_one_missing_endpoint_does_not_erase_a_whole_string(self):
        nodes = [n for n in mesh() if not (n.get('side') == 'top' and n['column'] == 1)]
        nodes = [n for n in nodes if not (n['kind'] == 'interior' and n['row'] == n['column'] == 1)]
        self.assertEqual(len(missing_crossings(nodes, [101, 101], 2)), 1)


if __name__ == '__main__':
    unittest.main()
