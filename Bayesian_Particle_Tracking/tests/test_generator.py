import unittest
import numpy as np
from unittest import TestCase
from Bayesian_Particle_Tracking import generate_data
from Bayesian_Particle_Tracking import io

class Test_generator(TestCase):
    def test_generator_length_correct(self):
        nsteps, sigma, mu, a, initial_coordinate = 100, 1, 1, 1, (0,0,0)
        test_data = generate_data.generator(nsteps, sigma, (mu, a), initial_coordinate)
        self.assertTrue(len(test_data) == 100)

    def test_generator_dimension_correct(self):
        nsteps, sigma, mu, a, initial_coordinate = 100, 1, 1, 1, (0,0,0)
        test_data_1 = generate_data.generator(nsteps, sigma, (mu, a), initial_coordinate, ndim = 1)
        self.assertTrue(test_data_1.shape[1] == 3)
        test_data_2 = generate_data.generator(nsteps, sigma, (mu, a), initial_coordinate, ndim = 2)
        self.assertTrue(test_data_2.shape[1] == 4)
        test_data_3 = generate_data.generator(nsteps, sigma, (mu, a), initial_coordinate, ndim = 3)
        self.assertTrue(test_data_3.shape[1] == 5)

if __name__ == '__main__':
    unittest.main()