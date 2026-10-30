import unittest

import numpy as np

from complete_edge_nodes import complete_endpoints


class CompleteEndpointTests(unittest.TestCase):
    def test_accepted_is_preserved_and_closest_peak_source_is_used(self):
        profile=np.zeros(120);cross=np.zeros(120)
        profile[[20,59,101]]=[.8,.5,.6]
        cross[60]=.1
        accepted=[{'profile_center':21.,'edge_id':'top-1'}]
        records=complete_endpoints([20,60,100],accepted,profile,cross)
        by_line={r['line_index']:r for r in records}
        self.assertEqual(len(records),3)
        self.assertEqual(by_line[0]['profile_center'],21)
        self.assertEqual(by_line[0]['origin'],'accepted')
        self.assertEqual(by_line[1]['profile_center'],60)
        self.assertEqual(by_line[1]['origin'],'cross_score')
        self.assertEqual(by_line[2]['profile_center'],101)
        self.assertEqual(by_line[2]['origin'],'edge_score')

    def test_no_peak_raises_instead_of_inventing_an_endpoint(self):
        with self.assertRaises(ValueError):
            complete_endpoints([20,60,100],[],np.zeros(120),np.zeros(120))


if __name__=='__main__':
    unittest.main()
