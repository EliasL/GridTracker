import unittest

import numpy as np

from local_node_deviations import group_deviations,local_outlier_ids


class LocalDeviationTests(unittest.TestCase):
    def test_two_groups_have_their_own_mean_and_sd(self):
        nodes=[{'id':i,'x':x,'y':20} for i,x in enumerate([10,11,12,70,71,72])]
        _,_,minima,groups,records=group_deviations(nodes,'x',100,2)
        self.assertEqual(len(minima),1)
        self.assertEqual([g['count'] for g in groups],[3,3])
        np.testing.assert_allclose([g['mean'] for g in groups],[11,71])
        np.testing.assert_allclose([g['population_std'] for g in groups],np.sqrt(2/3))
        self.assertEqual(len(records),6)
        for group in groups:
            z=np.array([r['signed_z'] for r in records if r['group']==group['group']])
            self.assertAlmostEqual(z.mean(),0)
            self.assertAlmostEqual(np.mean(z*z),1)

    def test_zero_variance_is_undefined_and_wrong_direction_is_excluded(self):
        nodes=[{'id':1,'x':10,'y':20},{'id':2,'x':10,'y':30},
               {'id':3,'x':70,'y':40,'scan_axes':['y']}]
        _,_,_,groups,records=group_deviations(nodes,'x',100,2)
        self.assertEqual(len(groups),1)
        self.assertEqual(len(records),2)
        self.assertTrue(all(r['signed_z'] is None for r in records))
        self.assertTrue(all(r['status']=='zero_group_variance' for r in records))

    def test_two_sd_rule_is_inclusive_and_combines_directions(self):
        records=[{'id':1,'distance_in_sd':2.0},{'id':2,'distance_in_sd':1.99},
                 {'id':1,'distance_in_sd':0.1},{'id':3,'distance_in_sd':None}]
        self.assertEqual(local_outlier_ids(records),{1})


if __name__=='__main__':
    unittest.main()
