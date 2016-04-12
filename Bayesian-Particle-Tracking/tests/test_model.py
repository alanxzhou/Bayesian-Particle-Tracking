from unittest import TestCase

import model

"""
Not really sure what do with this test.
"""

class TestModel(TestCase):
	def model_is_number(self):
		n = model.model(1, 1, 1, T = 300, center = 0, tau = 1)
		self.assertTrue(isinstance(n, float))