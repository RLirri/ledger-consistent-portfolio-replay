import unittest
from itertools import product

import numpy as np

from ledger_consistent_etf_trading.v2 import factorial


class Factorial(unittest.TestCase):
    def test_reconstruction_identity_on_arbitrary_non_monotone_cells(self):
        rng = np.random.default_rng(3)
        for k in (2, 4):
            cells = {c: float(rng.normal()) for c in product((0, 1), repeat=k)}
            self.assertLess(factorial.reconstruction_error(cells, k), 1e-12)
            self.assertEqual(len(factorial.effects(cells, k)), 2 ** k)

    def test_known_effects(self):
        # y = 10 + 3*A - 2*B + 5*A*B in 0/1 levels
        cells = {(a, b): 10 + 3 * a - 2 * b + 5 * a * b for a, b in product((0, 1), repeat=2)}
        self.assertAlmostEqual(factorial.interaction_2x2(cells[0, 0], cells[1, 0], cells[0, 1], cells[1, 1]), 5.0)
        beta = factorial.effects(cells, 2)
        self.assertAlmostEqual(beta[(0, 1)] * 4, 5.0)            # interaction in -1/+1 coding is I/4
        self.assertAlmostEqual(beta[()], np.mean(list(cells.values())))

    def test_extremes_need_not_bracket(self):
        cells = {(0, 0): 0.0, (1, 0): 4.0, (0, 1): -3.0, (1, 1): 1.0}     # intermediate cells lie outside [all-off, all-on]
        self.assertLess(factorial.reconstruction_error(cells, 2), 1e-12)

    def test_incomplete_design_rejected(self):
        with self.assertRaises(ValueError):
            factorial.effects({(0, 0): 1.0, (1, 1): 2.0}, 2)


if __name__ == "__main__":
    unittest.main()
