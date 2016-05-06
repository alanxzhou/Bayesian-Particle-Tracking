import unittest
import numpy as np
from unittest import TestCase
from Bayesian_Particle_Tracking import model
from Bayesian_Particle_Tracking import io

class TestModel(TestCase):
    def test_prior_positive_parameters(self):
        theta = (2e-10)
        self.assertTrue(model.log_prior(theta) !=  -np.inf)
        self.assertTrue(model.log_likelihood(theta, io.get_example_model('test_data.npy')) != -np.inf)

    def test_log_likelihood(self):
        D = 2e-10
        data = io.get_example_model('test_data.npy')
        self.assertTrue(round(model.log_likelihood(D, data)) == 88024)

    def test_displacement(self):
        data = io.get_example_model('test_data.npy')
        displacement_test = model.displacement(data.data).sum()
        self.assertTrue(round(displacement_test, 2) == 0.29)

    def test_log_prior(self):
        D = 1e-10
        test_value_jeff = model.log_prior(D)
        test_value_uniform = model.log_prior(D, prior = "Uniform")
        self.assertTrue(round(test_value_jeff, 2) == 20.81)
        self.assertTrue(test_value_uniform == 0)

    def test_log_posterior(self):
        D = 1e-10
        Dbad = 1e-6
        data = io.get_example_model('test_data.npy')
        testposterior = model.log_posterior(D, data)
        testposterior_bad = model.log_posterior(Dbad, data)
        self.assertTrue(round(testposterior) == 86028)
        self.assertTrue(testposterior_bad == -np.inf)

if __name__ == '__main__':
    unittest.main()
