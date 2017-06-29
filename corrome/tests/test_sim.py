import unittest
import numpy as np
import numpy.testing as npt
from numpy.random import RandomState
from skbio.stats.composition import closure
from corrome.sim import (chain_interactions, multinomial_sample,
                         compositional_noise, count_noise,
                         train_count_parameters,
                         train_compositional_parameters)


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
                         8.76415025e-03]]).T
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
        cov = np.eye(2)
        res = compositional_noise(cov, 5)
        exp = np.array([[0.79460719, 0.06556888, 0.13982394],
                        [0.77959817, 0.19531899, 0.02508284],
                        [0.51151606, 0.03646136, 0.45202258],
                        [0.53313183, 0.13909273, 0.32777544],
                        [0.35615769, 0.4121334 , 0.23170892]])
        npt.assert_allclose(res, exp)

    def test_train_count_parameters(self):
        data = np.array([[18, 25, 20, 13, 25],
                         [10, 21, 16, 17, 39],
                         [17, 28, 25,  9, 19],
                         [14, 28, 15, 11, 30],
                         [15, 27, 32, 13, 40]])
        lam, p = train_count_parameters(data)
        self.assertAlmostEqual(lam, 105.40000000000001)
        exp_p = np.array([0.14041746, 0.24478178, 0.20493359,
                          0.11954459, 0.29032258])
        npt.assert_allclose(exp_p, p)

    def test_train_compositional_parameters(self):
        cov = np.eye(2)
        # generate model with identify covariance and 0 mean.
        data = compositional_noise(cov, 1000)
        mu, cov = train_compositional_parameters(data)

        exp_mu = np.array([-0.02173648, -0.00990328])
        npt.assert_allclose(exp_mu, mu, atol=1e-5)
        exp_cov = np.array([[0.95037693, -0.01333414],
                            [-0.01333414, 0.96476946]])
        npt.assert_allclose(exp_cov, cov, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
