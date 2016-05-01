import numpy as np
from numpy.random import normal
from numpy.random import uniform
import scipy
from Bayesian_Particle_Tracking.printable import Printable
from Bayesian_Particle_Tracking.prior import JeffreysPrior
from Bayesian_Particle_Tracking.prior import UniformPrior

def displacement(data):
    """
    This returns the displacement between successive points for a data set of positions.
        data: size 3 x n array
            positional data input in cartesian coordinates (x,y,z)
    """
    #point_before is the data list minus the last element.
    #data_points is the data list minus the first element.
    #distance is array containing the displacements between two consecutive points in the series.
    #Using np.diff seems to run a lot slower than the following 5 lines:
    data_length = len(data)
    point_before = data[:data_length-1]
    x_before, y_before, z_before = point_before[:,0], point_before[:,1], point_before[:,2]
    data_points = data[1:]
    x_data, y_data, z_data = data_points[:,0], data_points[:,1], data_points[:,2]
    distance = np.sqrt((x_before-x_data)**2+(y_before-y_data)**2+(z_before-z_data)**2)
    return distance

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
    '''
    def __init__(self, data, nwalkers = 1):
        self.data = data
        self.n = len(data)
        self.initial_position = data[0]
        self.x = data[:,0]
        self.y = data[:,1]
        self.z = data[:,2]
        self.sigma = data[:,3]
    
    #Allows translation of the object
    def translate(self, offset):
        return diffusion(np.array((self.x+offset, self.y+offset, self.z+offset, self.sigma)).T)


def log_prior(theta, lower_bound = 1e-12, upper_bound = 1e-8, prior = "Jeffreys"):
    """
    Log prior function for 3D diffusion process of a single particle.
    Default prior is a Jeffreys prior with lower bound at 1e-12 and upper bound at 1e-8

    Parameters
    ----------
        D: diffusion coefficient
        lower_bound: lower_bound of Prior
        upper_bound: upper_bound of Prior
    """
    if prior == "Jeffreys":
        theta_prior = JeffreysPrior(lower_bound, upper_bound).lnprob(theta)
    elif prior == "Uniform":
        theta_prior = UniformPrior(lower_bound, upper_bound).lnprob(theta)
    else:
        raise ValueError("Prior not recognized. ")
    return theta_prior

def log_likelihood(theta, diffusion_object, tau = 1, unknown = 'D', known_variables = None):
    """
    Likelihood function for 3D diffusion process of a single particle.

    Parameters
    ----------
        diffusion_object: contains the following:
            data: positional data: assumes data is in the form of column vectors (traj_x,traj_y,traj_z)
                where traj_x,traj_y,traj_z are the coordinate positions of the particle
            sigma: variance on positional data; we assume the uncertainty is Gaussian; true dependency will be from raw image data
        theta: unknown parameter;
        unknown: The unknown parameter to be determined. Default is D. Must be input as string. Possible parameters are:
            D: diffusion coefficient
            a: radius of particle
            mu: dynamic viscosity of medium
            T: temperature of medium
        known_variables: Given as a tuple (a,b), where a and b are known values for the two unknown parameters if the uknown parameter is not D.
            For the following unknown parameters, the tuple (a,b) should be given as follows:
                a: (mu, T)
                mu: (a, T)
                T: (a, mu)
    """
    if unknown == 'D':
        D = theta
    elif unknown != 'D':
        kb = 1.38e-23
        if unknown == 'a':
            a = theta
            mu, T = known_variables
            D = (kb*T)/(6*np.pi*mu*a)
        elif unknown == 'mu':
            mu = theta
            a, T = known_variables
            D = (kb*T)/(6*np.pi*mu*a)
        elif unknown == 'T':
            T = theta
            a, mu = known_variables
            D = (kb*T)/(6*np.pi*mu*a)
        elif isinstance(unkonwn, str):
            raise ValueError('%s is not a valid parameter. Valid parameters are D, a, mu, T.' %unknown)
        else:
            raise TypeError('%s is not a string. Valid parameters are D, a, mu, T.' %unknown)

    data = diffusion_object.data
    sigma = diffusion_object.sigma
    distance = displacement(data)
    
    #Delete the first element of the sigma array to match array sizes
    sigma = sigma[1:len(sigma)]
    
    diffusion_factor = np.sqrt(6*D*tau)

    result = (-len(data)/2)*np.log(2*np.pi)+np.sum(np.log((diffusion_factor**2+sigma**2)**(-1/2)))+np.sum(-((distance)**2)/(2*(diffusion_factor**2+sigma**2)))
    return result
    
def log_posterior(theta, diffusion_object, tau = 1, unknown = 'D', known_variables = None, lower_bound = 1e-12, upper_bound = 1e-8, prior = "Jeffreys"):
    """
    Log posterior function for 3D diffusion process of a single particle. 

    Parameters are given by parameters in log_likelihood() function and log_prior() function
        theta, diffusion_object, tau, unknown, known_variables given are parameters of log_likelihood()
        lower_bound, upper_bound, Jeffreys given are parameters of log_prior function

    """
    prior = log_prior(theta, lower_bound = lower_bound, upper_bound = upper_bound, prior = "Jeffreys")
    if prior == -np.inf:
        return prior
    return prior + log_likelihood(theta, diffusion_object, tau = tau, unknown = unknown, known_variables = known_variables)

