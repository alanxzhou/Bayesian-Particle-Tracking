import unittest
import numpy as np
from unittest import TestCase
from Bayesian_Particle_Tracking import model
from Bayesian_Particle_Tracking import io 
from Bayesian_Particle_Tracking.io import input_data

"""
Unit tests for model
"""

class TestModel(TestCase):
    def test_model_is_number(self):
        data = list(input_data.data)
        self.assertTrue(isinstance(data, list))

    def test_model_object_works(self):
        testdata = io.input_data
        self.assertTrue(isinstance(testdata.n, int))
        self.assertTrue(len(testdata.data[0])==4)

    def test_model_positive_parameters(self):
        theta = (10**(-4), 300, 10**(-8))
        self.assertTrue(model.log_prior(theta) != -np.inf)
        self.assertTrue(model.log_likelihood(theta, io.input_data) != -np.inf)

if __name__ == '__main__':
    unittest.main()
