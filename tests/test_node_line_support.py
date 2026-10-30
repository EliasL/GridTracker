import unittest

import numpy as np

from experiments.node_line_support import node_line_scores, summarize


class LineSupportTests(unittest.TestCase):
    def test_scores_exclude_self_and_respect_direction(self):
        nodes=[{'id':1,'x':10,'y':10}, {'id':2,'x':10,'y':50},
               {'id':3,'x':10,'y':0,'scan_axes':['x']}]
        scores,average=node_line_scores(nodes,2)
        np.testing.assert_allclose(scores[:,0],[2,2,2])
        np.testing.assert_allclose(scores[:2,1],np.exp(-.5*(40/2)**2))
        self.assertTrue(np.isnan(scores[2,1]))
        np.testing.assert_allclose(average,[1,1,2])

    def test_singleton_has_zero_support_without_special_case(self):
        scores,average=node_line_scores([{'id':7,'x':12,'y':20}],3)
        np.testing.assert_array_equal(scores,[[0,0]])
        np.testing.assert_array_equal(average,[0])

    def test_population_statistics(self):
        stats=summarize([1,1,3,3])
        self.assertEqual(stats['mean'],2)
        self.assertEqual(stats['population_std'],1)
        self.assertEqual(stats['lower'],1)
        self.assertEqual(stats['upper'],3)


if __name__=='__main__':
    unittest.main()
