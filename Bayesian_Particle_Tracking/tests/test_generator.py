import unittest
import numpy as np
from unittest import TestCase
from Bayesian_Particle_Tracking import generate_data
from Bayesian_Particle_Tracking import io

class TestModel(TestCase):
    def test_generator_length_correct(self):
        nsteps, sigma, mu, a, initial_coordinate = 100, 1, 1, 1, (0,0,0)
        test_data = generate_data.generator(nsteps, sigma, mu, a, initial_coordinate)
        self.assertTrue(len(test_data) == 100)

if __name__ == '__main__':
    unittest.main()