import unittest
from unittest import TestCase
from Bayesian_Particle_Tracking import model
from Bayesian_Particle_Tracking import io

"""
Unit tests for model
"""

class TestModel(TestCase):
	def test_model_is_number(self):
		n = list(model.generator(1, 1, 1, 1, [0,0,0],T = 300, center = 0, tau = 1))
		self.assertTrue(isinstance(n, list))

	def test_model_works(self):
		testdata = io.input_data
		self.assertTrue(isinstance(testdata.n, int))
		self.assertTrue(len(testdata.data[0])==4)

if __name__ == '__main__':
	unittest.main()
