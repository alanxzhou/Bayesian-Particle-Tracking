import numpy as np
from numpy.random import normal
from numpy.random import uniform
import matplotlib as plt
import scipy

def model(nsteps, sigma, mu, a, T = 300, nwalkers = 1, center = 0, tau = 1):
	"""
	This function provides the trajectory for a 3D diffusion process.
	Returns trajectories in (x,y,z) coordiantes as column vectors

	Parameters:
		nsteps: number of steps to take
		sigma: uncertainty in positional data; is a function of parameters in particle tracking in raw data
		mu: viscosity of medium
		a: radius of particle
		T: temperature (default at room temperature ~ 300K)
		tau: time constant (default at 1s)
		nwalkers: number of walkers (NOTE: not implemented yet)

	Note that D is the diffusion constant and is a funciton of a, mu, T
	"""
	
	kb = 1.38*10**(-23)
	D = (kb*T)/(6*np.pi*mu*a)
	sigma1 = np.sqrt(3*D*tau)
	r = abs(normal(center, np.sqrt(sigma1**2+sigma**2), nsteps))/2
	theta = uniform(0,np.pi,nsteps)
	phi = uniform(0,2*np.pi,nsteps)
	#Spherical to Cartesian Coordinates
	x = r*np.sin(theta)*np.cos(phi)
	y = r*np.sin(theta)*np.sin(phi)
	z = r*np.cos(theta)
	#Trajectories
	traj_x = np.cumsum(x)
	traj_y = np.cumsum(y)
	traj_z = np.cumsum(z)
	return(np.array((traj_x, traj_y, traj_z)).T)

def likelihood(data, sigma, mu, a, T = 300, center = 0, tau = 1):
	"""
	Likelihood function for 3D diffusion process of a single particle.

	Parameters:
		data: positional data: assumes data is in the form of column vectors (traj_x,traj_y,traj_z)
			where traj_x,traj_y,traj_z are the coordinate positions of the particle
		sigma: variance on positional data; we assume the uncertainty is Gaussian; true dependency will be from raw image data
		mu: viscosity of medium
		a: radius of particle
		T: temperature
		tau: time constant (default at 1s)
		D: diffusion constant; function of a, mu, T
	"""
	kb = 1.38*10**(-23)
	D = (kb*T)/(6*np.pi*mu*a)
	sigma1 = np.sqrt(3*D*tau)

	#This should give us the delta x_{i,i-1}. point_before is just the data list minus the last element.
	#data_points is the data list minus the first element.
	#point_before is the data list minus the last element.
	#distance is the measurement of the distance between two consecutive points in the series.
	point_before = list(data)
	del point_before[len(point_before)-1]
	point_before = np.array(point_before)
	x_before, y_before, z_before = point_before[:,0], point_before[:,1], point_before[:,2]
	data_points = list(data)
	del data_points[0]
	data_points = np.array(data_points)
	x_data, y_data, z_data = data_points[:,0], data_points[:,1], data_points[:,2]
	distance = np.sqrt((x_before-x_data)**2+(y_before-y_data)**2+(z_before-z_data)**2)

	return (2*np.pi)**(-len(data)/2)*np.prod((sigma1**2+sigma**2)**(-1/2))*np.exp((-((distance)**2)/(2*(sigma1**2+sigma**2))).sum())


