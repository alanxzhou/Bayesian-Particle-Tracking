import numpy as np
from numpy.random import normal
from numpy.random import uniform
import scipy
import os

"""
Generates data to test against the model.

Original test data created with the following parameters:
nsteps = 1000, T = 300, mu = 10^-4, a = 10^-8, intial_coordinate = (0,0,0)
(D = 2.19*10^-10)
"""

def data_generation(a, b, c, d, e, f):
    
    data = generator(a, b, c, d, e, f)
    np.save('test_data', data)


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