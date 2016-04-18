#Mostly from Tom Dimiduk at https://github.com/p201-sp2016/package-example/blob/master/package_example/prior.py

from Bayesian_Particle_Tracking.printable import Printable
import numpy as np

class UniformPrior(Printable):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def lnprob(self, p):
        if self.lower_bound < p < self.upper_bound:
            return 0
        else:
            return -np.inf

class JeffreysPrior(Printable):
	def __init__(self, lower_bound, upper_bound):
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound

	def lnprob(self, p):
		if self.lower_bound < p < self.upper_bound:
			return np.log(1/(p*np.log(self.upperbound/self.lower_bound)))
		else:
			return -np.inf