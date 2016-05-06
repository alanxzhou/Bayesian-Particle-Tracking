import unittest
import numpy as np
from unittest import TestCase
from Bayesian_Particle_Tracking import model
from Bayesian_Particle_Tracking import io

class Test_io(TestCase):
    def test_data(self):
        data = list(io.get_example_model('test_data.npy').data)
        self.assertTrue(isinstance(data, list)),

    def test_length(self):
        testdata = io.get_example_model('test_data.npy')
        self.assertTrue(len(testdata.data[0])==5)

if __name__ == '__main__':
    unittest.main()
