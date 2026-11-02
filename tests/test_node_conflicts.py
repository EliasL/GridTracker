import unittest

from node_conflicts import resolve_conflicts


class ConflictTests(unittest.TestCase):
    def test_local_bending_beats_a_rigid_periodic_guess(self):
        result = dict(width=4, maxima=dict(x=dict(clean=[20, 70, 120]), y=dict(clean=[20, 70])))
        nodes = [dict(id=1, kind='interior', row=0, column=0, x=20., y=24.),
                 dict(id=2, kind='interior', row=0, column=2, x=120., y=24.),
                 dict(id=3, kind='interior', row=1, column=1, x=75., y=70.),
                 dict(id=4, kind='interior', row=0, column=1, x=75., y=24.),
                 dict(id=5, kind='interior', row=0, column=1, x=70., y=30.)]
        kept, removed = resolve_conflicts(nodes, result)
        self.assertIn(4, [n['id'] for n in kept])
        self.assertEqual([n['id'] for n in removed], [5])
        self.assertEqual(removed[0]['retained_id'], 4)
        nodes[-1]['human_edited'] = True
        kept, removed = resolve_conflicts(nodes, result)
        self.assertIn(5, [n['id'] for n in kept])
        self.assertEqual(removed[0]['id'], 4)

    def test_two_incompatible_human_edits_are_not_silently_discarded(self):
        result = dict(width=4, maxima=dict(x=dict(clean=[20]), y=dict(clean=[20])))
        node = dict(id=1, kind='interior', row=0, column=0, x=20., y=20., human_edited=True)
        with self.assertRaisesRegex(ValueError, 'Two human edits'):
            resolve_conflicts([node, dict(node, id=2, x=24.)], result)

