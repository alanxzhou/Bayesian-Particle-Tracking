import unittest
import numpy as np
from scipy.optimize import curve_fit
from unittest import TestCase
from Bayesian_Particle_Tracking.model import diffusion
from Bayesian_Particle_Tracking import data_analysis
from Bayesian_Particle_Tracking import io

class Test_data_analysis(TestCase):
    def test_CGW(self):
        data = diffusion(io.get_example_model('test_data.npy').data)
        msd, sigma, counter = data_analysis.CGW_analysis(data, 1000, down_sample = 1)
        popt, pcov = curve_fit(data_analysis.line_fit3D, counter, msd, sigma = sigma)
        D = popt[0]
        sigma_D = np.sqrt(pcov[0])
        self.assertTrue(round(D*1e10,2) == 2.25)
        self.assertTrue(round(sigma_D.sum()*1e13,2) == 3.01)

    def test_MLE(self):
        data = diffusion(io.get_example_model('test_data.npy').data)
        D, Dbest, loglike, Dmin, Dmax = data_analysis.max_likelihood_estimation(data, -11, -9, 100)
        self.assertTrue(round(Dbest*1e10,2) == 2.15)

if __name__ == '__main__':
    unittest.main()