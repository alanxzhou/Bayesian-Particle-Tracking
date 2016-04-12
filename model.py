import numpy as np
from numpy.random import normal
from numpy.random import uniform
import matplotlib as plt
import scipy

def trajectory(nsteps, sigma, mu, a, T = 300, nwalkers = 1, center = 0, tau = 1):
	"""
	This function provides the trajectory for a 3D random walk of n=nsteps steps.
	"""
	
	kb = 1.38*10**(-23)
	D = (kb*T)/(6*np.pi*mu*a)
	sigma1 = np.sqrt(3*D*tau)
	r = abs(normal(center, np.sqrt(sigma1**2+sigma**2), nsteps))/2
	theta = uniform(0,np.pi,nsteps)
	phi = uniform(0,2*np.pi,nsteps)
	x = r*np.sin(theta)*np.cos(phi)
	y = r*np.sin(theta)*np.sin(phi)
	z = r*np.cos(theta)
	traj_x = np.cumsum(x)
	traj_y = np.cumsum(y)
	traj_z = np.cumsum(z)
	return(traj_x, traj_y, traj_z)

def model(sigma, mu, a, T = 300, center = 0, tau = 1):
	"""
	Generates the length of the next jump for a 3D random walk. 
	Note that sigma as an uncertainty is dependent on the parameters of the raw data.

	Parameters:
		sigma: uncertainty in data measurement
		mu: dynamic viscosity of fluid
		a: radius of particle
		T: temperature (default at 300K, room temperature)
		center: normal distribution centered at 0
		tau: time constant (default at 1s)
		D: diffusion constant; function of a, mu, T
	"""
	kb = 1.38*10**(-23)
	D = (kb*T)/(6*np.pi*mu*a)
	sigma1 = np.sqrt(3*D*tau)
	r = abs(normal(center,np.sqrt(sigma1**2+sigma**2)))/2
	return(r)


def likelihood(data, sigma, mu, a, T = 300, center = 0, tau = 1):
	"""
	Likelihood function for a 3D random walk. 

	Parameters:
		data: positional data: assumes data is in the form of delta_r where r is length of the step
			if data is in 
		sigma: variance on positional data (may be part of the data array: this may be integrated into data parameter)
		mu: viscosity of medium
		a: radius of particle
		T: temperature
		tau: time constant (default at 1s)
		D: diffusion constant; function of a, mu, T
	"""
	kb = 1.38*10**(-23)
	D = (kb*T)/(6*np.pi*mu*a)
	sigma1 = np.sqrt(3*D*tau)
	#predicted = model(sigma, mu, a, T = T, center = center, tau = tau)

	#This should give us the delta x_{i,i-1}. point_before is just the data list minus the last element.
	#data_points is the data list minus the first element.
	point_before = list(old_list)
	point_before.pop[len(point_before)]
	point_before = np.array(point_before)
	data_points = list(data)
	data_points.pop[0]
	data_points = np.array(data_points)


	return (2*np.pi)**(-len(data)/2)*np.prod((sigma1**2+sigma**2)**(-1/2))*np.exp((-((data_points-point_before)**2)/(2*(sigma1**2+sigma**2))).sum())

