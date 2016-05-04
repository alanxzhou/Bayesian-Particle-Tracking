import numpy as np
from numpy.random import normal
from numpy.random import uniform
import os

def data_generation(nsteps, sigma, theta, initial_coordinate, T = 300, tau = 1, parameter = None, ndim = 3):
    """
    Saves generated data from the generator function as 'test_data' in the current directory.

    Parameters are the same as the generator functino.
    """
    data = generator(nsteps, sigma, theta, initial_coordinate, T = T, tau = tau, parameter = None, ndim = 3)
    np.save('test_data', data)


def generator(nsteps, sigma, theta, initial_coordinate, T = 300, tau = 1, parameter = None, ndim = 3):
    """
    This function provides the trajectory for a 3D diffusion process.
    Returns trajectories in (x,y,z) coordinates as column vectors
    
    Parameters
    ----------
    nsteps: number of steps to take
    sigma: uncertainty in positional data; is a function of parameters in particle tracking in raw data
    theta: input parameters
        mu: viscosity of medium
        a: radius of particle
        D: diffusion coefficient (this selection is only if the value of parameter is "D")
    initial_coordinate: initial coordinate of particle in array [x,y,z]
    T: temperature (default at room temperature ~ 300K)
    tau: time constant (default at 1s)
    parameter: choose parameter with which to generate data. Options:
        "D": generate random walk with D as the input parameter for theta
    ndim: number of dimensions to generate
    
    Note that D is the diffusion coefficient and is a function of a, mu, T given by the Stokes-Einsten Equation
    """
    x_init, y_init, z_init = initial_coordinate

    if parameter == "D":
        D = theta
    else:
        mu, a = theta
        kb = 1.38*10**(-23)
        D = (kb*T)/(2*ndim*np.pi*mu*a)

    if not isinstance(ndim, int):
        raise TypeError('%s is not an integer.' %ndim)
    else:
        if ndim < 1 or ndim > 3:
            raise ValueError('%s is invalid. Particle Tracking is for 1, 2, or 3 dimensions' %ndim)

    r_noise = normal(0, sigma, nsteps)
    theta_noise = uniform(0,np.pi,nsteps)
    phi_noise = uniform(0,2*np.pi,nsteps)

    diffusion_factor = np.sqrt(2*ndim*D*tau)
    r = normal(0, diffusion_factor, nsteps)
    theta = uniform(0,np.pi,nsteps)
    phi = uniform(0,2*np.pi,nsteps)

    sigmaarray = np.ones(nsteps)*sigma
    tau_array = np.ones(nsteps)*tau

    if ndim == 1:
        x_noise = r_noise
        x = r

        traj_x = np.cumsum(x) + x_noise + x_init
        return(np.array((traj_x, sigmaarray, tau_array)).T)
    elif ndim == 2:
        x_noise = r_noise*np.cos(phi_noise)
        y_noise = r_noise*np.sin(phi_noise)
        x = r*np.cos(phi)
        y = r*np.sin(phi)

        traj_x = np.cumsum(x) + x_noise + x_init
        traj_y = np.cumsum(y) + y_noise + y_init
        return(np.array((traj_x, traj_y, sigmaarray, tau_array)).T)
    elif ndim == 3:
        x_noise = r_noise*np.sin(theta_noise)*np.cos(phi_noise)
        y_noise = r_noise*np.sin(theta_noise)*np.sin(phi_noise)
        z_noise = r_noise*np.cos(theta_noise)
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)

        traj_x = np.cumsum(x) + x_noise + x_init
        traj_y = np.cumsum(y) + y_noise + y_init
        traj_z = np.cumsum(z) + z_noise + z_init

        return(np.array((traj_x, traj_y, traj_z, sigmaarray, tau_array)).T)