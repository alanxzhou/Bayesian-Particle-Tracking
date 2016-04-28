import unittest
import numpy as np
from unittest import TestCase
from Bayesian_Particle_Tracking import model
from Bayesian_Particle_Tracking import io

"""
Unit tests for model
"""

class TestModel(TestCase):
    def test_model_is_number(self):
        data = list(io.get_example_model('test_data.npy').data)
        self.assertTrue(isinstance(data, list))

    def test_model_object_works(self):
        testdata = io.get_example_model('test_data.npy')
        self.assertTrue(isinstance(testdata.n, int))
        self.assertTrue(len(testdata.data[0])==4)

    def test_model_positive_parameters(self):
        theta = (2*10**(-10))
        self.assertTrue(model.log_prior(theta) !=  -np.inf)
        self.assertTrue(model.log_likelihood(theta, io.get_example_model('test_data.npy')) != -np.inf)

if __name__ == '__main__':
    unittest.main()
