import unittest

import numpy as np

from fit_bezier_curves import bezier_basis, fit_curve


class BezierTests(unittest.TestCase):
    def test_cubic_recovery(self):
        along = np.linspace(0,100,20)
        true = np.array([20.,35,35,20])
        values = bezier_basis(along/100)@true
        exact, error = fit_curve(along,values,101,101,25)
        np.testing.assert_allclose(exact,true,atol=1e-4)
        self.assertLess(error,1e-4)

    def test_underconstrained_unregularized_curves_raise(self):
        for along, across in (([30],[27]), ([20,80],[22,28]), ([0,50,100],[20,30,20])):
            with self.assertRaises(ValueError):
                fit_curve(along,across,101,101,25)


if __name__=='__main__':
    unittest.main()
