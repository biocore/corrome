import unittest
import numpy as np
import numpy.testing as npt
from numpy.random import RandomState
from skbio.stats.composition import closure
from corrome.sim import (chain_interactions, multinomial_sample,
                         compositional_noise, count_noise,
                         train_count_parameters,
                         train_composition_parameters)


class TestSim(unittest.TestCase):

    def setUp(self):
        pass

    def test_chain_interactions(self):

        gradient = np.linspace(0, 15, 10)
        mu = [5, 10]
        sigma = [2, 2]

        res = chain_interactions(gradient, mu, sigma)
        exp = np.array([[8.76415025e-03, 4.97385694e-02, 1.40955938e-01,
                         1.99471140e-01, 1.40955938e-01, 4.97385694e-02,
                         8.76415025e-03, 7.71139498e-04, 3.38815049e-05,
                         7.43359757e-07],
                        [7.43359757e-07, 3.38815049e-05, 7.71139498e-04,
                         8.76415025e-03, 4.97385694e-02, 1.40955938e-01,
                         1.99471140e-01, 1.40955938e-01, 4.97385694e-02,
                         8.76415025e-03]])
        npt.assert_allclose(res, exp)

    def test_multinomial_sample(self):
        rng = RandomState(0)
        X = np.array([[8.76415025e-03, 4.97385694e-02, 1.40955938e-01,
                       1.99471140e-01, 1.40955938e-01, 4.97385694e-02,
                       8.76415025e-03, 7.71139498e-04, 3.38815049e-05,
                       7.43359757e-07],
                      [7.43359757e-07, 3.38815049e-05, 7.71139498e-04,
                       8.76415025e-03, 4.97385694e-02, 1.40955938e-01,
                       1.99471140e-01, 1.40955938e-01, 4.97385694e-02,
                       8.76415025e-03]])
        X = closure(X)
        lam = 5
        res = multinomial_sample(X, lam, rng)
        exp = np.array([[0, 2, 3, 3, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 3, 1, 1, 0, 0]])
        npt.assert_allclose(res, exp)

    def test_compositional_noise(self):
        pass

    def test_count_noise(self):
        pass

    def test_train_count_parameters(self):
        pass

    def test_train_compositional_parameters(self):
        pass


if __name__ == "__main__":
    unittest.main()
