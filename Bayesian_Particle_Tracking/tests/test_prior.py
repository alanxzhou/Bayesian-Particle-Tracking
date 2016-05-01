import unittest
import numpy as np
from unittest import TestCase
from Bayesian_Particle_Tracking import prior

class TestModel(TestCase):
    def test_prior_uniform(self):
        lower_bound, upper_bound, = 0, 1
        test_uniform = prior.UniformPrior(lower_bound, upper_bound)
        pwork, pfail = 0.5, -0.5
        self.assertTrue(test_uniform.lnprob(pwork) == 0)
        self.assertTrue(test_uniform.lnprob(pfail) == -np.inf)

    def test_prior_jeffreys(self):
        lower_bound, upper_bound, = 0.1, 1
        test_jeffreys = prior.JeffreysPrior(lower_bound, upper_bound)
        pwork, pfail = 0.5, -0.5
        self.assertTrue(round(test_jeffreys.lnprob(pwork),3) == -0.141)
        self.assertTrue(test_jeffreys.lnprob(pfail) == -np.inf)

if __name__ == '__main__':
    unittest.main()