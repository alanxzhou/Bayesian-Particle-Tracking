from unittest import TestCase

import model

"""
Not really sure what to test / how to test things. This needs to be updated.
"""

class TestModel(TestCase):
	def model_is_number(self):
		n = model.model(1, 1, 1, T = 300, center = 0, tau = 1)
		self.assertTrue(isinstance(n, float))