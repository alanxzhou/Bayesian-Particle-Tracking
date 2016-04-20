import numpy as np
from numpy.random import normal
from numpy.random import uniform
import scipy
from Bayesian_Particle_Tracking.printable import Printable
from Bayesian_Particle_Tracking.prior import JeffreysPrior
from Bayesian_Particle_Tracking.prior import UniformPrior

def generator(nsteps, sigma, mu, a, initial_coordinate, T = 300, nwalkers = 1, center = 0, tau = 1):
	"""
	This function provides the trajectory for a 3D diffusion process.
	Returns trajectories in (x,y,z) coordinates as column vectors

	Parameters:
		nsteps: number of steps to take
		sigma: uncertainty in positional data; is a function of parameters in particle tracking in raw data
		mu: viscosity of medium
		a: radius of particle
		initial_coordinate: initial coordinate of particle in array [x,y,z]
		T: temperature (default at room temperature ~ 300K)
		tau: time constant (default at 1s)
		nwalkers: number of walkers (NOTE: not implemented yet)

	Note that D is the diffusion coefficient and is a function of a, mu, T
	"""
	
	x_start = normal(initial_coordinate[0],sigma*np.sqrt(3))
	y_start = normal(initial_coordinate[1],sigma*np.sqrt(3))
	z_start = normal(initial_coordinate[2],sigma*np.sqrt(3))
  
	kb = 1.38*10**(-23)
	D = (kb*T)/(6*np.pi*mu*a)
	sigma1 = np.sqrt(3*D*tau)
	r = normal(center, sigma1, nsteps)
	theta = uniform(0,np.pi,nsteps)
	phi = uniform(0,2*np.pi,nsteps)
	#Spherical to Cartesian Coordinates
	x = r*np.sin(theta)*np.cos(phi)
	y = r*np.sin(theta)*np.sin(phi)
	z = r*np.cos(theta)
	#Trajectories
	traj_x = np.cumsum(x) + x_start
	traj_y = np.cumsum(y) + y_start
	traj_z = np.cumsum(z) + z_start
	sigmaarray = np.ones(len(traj_x))*sigma
	return(np.array((traj_x, traj_y, traj_z, sigmaarray)).T)

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

	#Turn data into displacements instead of positions
	data = data-data[0]

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

	#This assumes a true initial coordinate without measurement uncertainty
	return (2*np.pi)**(-len(data)/2)*np.prod((sigma1**2+sigma**2)**(-1/2))*np.exp((-((distance)**2)/(2*(sigma1**2+sigma**2))).sum())

class diffusion(Printable):
    '''
    Contains data and relevent parameters for a 3-D Diffusion Process

    Attributes
    ----------
    data: size 3 x n array
        positional data input in cartesian coordinates (x,y,z)
    sigma: nonzero float
        specifies measurement uncertainty on measurement of particle position
    n : integer
        number of steps taken in diffusion process
    position: length 3 listlike
        initial position
    nwalkers: integer
        number of particles being tracked. TO BE IMPLEMENTED
    '''
    def __init__(self, data, nwalkers = 1):
        self.data = data
        self.n = len(data)
        self.position = data[0]
        self.x = data[:,0]
        self.y = data[:,1]
        self.z = data[:,2]
        self.sigma = data[:,3]
    
    def translate(self,offset):
        return diffusion(data+np.array(offset))

def log_prior(theta):
    mu, T, a = theta
    mu_prior = JeffreysPrior(10**(-6), 10**(-2)).lnprob(mu)
    T_prior = UniformPrior(273, 400).lnprob(T)
    a_prior = JeffreysPrior(10**(-10), 10**(-6)).lnprob(a)
    if mu > 0 and T > 0 and a > 0:
        return mu_prior+T_prior+a_prior
    else:
        return -np.inf

def log_likelihood(theta, diffusion_object, tau = 1):
    mu, T, a = theta
    data = diffusion_object.data
    sigma = diffusion_object.sigma
    
    sigma = list(sigma)
    del sigma[0]
    sigma = np.array(sigma)
    
    kb = 1.38*10**(-23)
    D = (kb*T)/(6*np.pi*mu*a)
    if D <= 0:
        return -np.inf
    sigma1 = np.sqrt(3*D*tau)
    
    #Turn data into displacements instead of positions
    data = data-data[0]

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

    #Should be correct with the correct log properties
    result = (-len(data)/2)*np.log(2*np.pi)+np.sum(np.log((sigma1**2+sigma**2)**(-1/2)))+(-((distance)**2)/(2*(sigma1**2+sigma**2))).sum()
    return result
    
def log_posterior(theta, diffusion_object):
    return log_prior(theta) + log_likelihood(theta, diffusion_object)

"""
The following functions are defined such that T is taken as a given number rather than an uknown parameter.
"""

def log_prior2(theta):
    mu, a = theta
    mu_prior = JeffreysPrior(10**(-6), 10**(-2)).lnprob(mu)
    a_prior = JeffreysPrior(10**(-10), 10**(-6)).lnprob(a)
    if mu > 0 and a > 0:
        return mu_prior+a_prior
    else:
        return -np.inf

def log_likelihood2(theta, diffusion_object, tau = 1, T = 300):
    mu, a = theta
    data = diffusion_object.data
    sigma = diffusion_object.sigma
    
    sigma = list(sigma)
    del sigma[0]
    sigma = np.array(sigma)
    
    kb = 1.38*10**(-23)
    D = (kb*T)/(6*np.pi*mu*a)
    if D <= 0:
        return -np.inf
    sigma1 = np.sqrt(3*D*tau)
    
    #Turn data into displacements instead of positions
    data = data-data[0]

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

    #Should be correct with the correct log properties
    result = (-len(data)/2)*np.log(2*np.pi)+np.sum(np.log((sigma1**2+sigma**2)**(-1/2)))+(-((distance)**2)/(2*(sigma1**2+sigma**2))).sum()
    return result
    
def log_posterior2(theta, diffusion_object):
    return log_prior2(theta) + log_likelihood2(theta, diffusion_object)
