# Bayesian Particle Tracking

Background: Accurate estimation of the diffusion coefficient and its uncertainty is important in adequately understanding any diffusion process. From a calculation of the diffusion coefficient, one can directly determine properties of the particle (size) or of the medium (dynamic viscosity). There does not seem to exist a standard accepted procedure for calculating the diffusion coefficient. Current methods are very ad hoc and can be improved upon. 

With a generative model for the data, the diffusion coefficient, and its uncertainty can be accurately measured in a number of ways. This package is designed to allow bayesian analysis of particle tracking data. Specifically, it allows estimations of the diffusion coefficient for a diffusion process in 3D particle tracking given a trajectory and the uncertainty on the measurments in that trajectory.

### Parameters:

sigma: uncertainty in measurement on position

mu: dynamic viscosity of the medium

a: radius of the particle

T: temperature

D: diffusion coefficient; this is a function of mu, a, and T. Specifically by Stokes-Einsten:
D = kb*T/(6*pi*mu*a)

### Data: 
The package tutorial uses simulated data to demonstrate its efficacy. This is because it is much easier to demonstrate efficacy when the desired answer is known.

The package also uses real data provided in a pandas data provided by Viva Horowitz.

## Installation

This package can be installed by cloning the repository via the terminal ("git clone https://github.com/p201-sp2016/Bayesian-Particle-Tracking") and then running "python setup.py install" inside the root directory.

## Files:

- Model.ipynb: a jupyter notebook containing the mathematical details of the statistical models and the appliaction of analyses to a short test data set, test_data.npy.

- Marginalization.ipynb: a jupyter notebook containing the preliminary python implementation of marginalization in order to estimate D. This is a work in progress, and once a workaround is found for the issues found in this notebook, this notebook will be deleted and merged with tutorial.ipynb.

- Tutorial.ipynb: a jupyter notebook containing analyses on longer simulated data, as well as real data provided by Viva Horowitz

- test_data.npy: a numpy array containing simulated test_data of 1000 positional and time coordinates in 3D.

- data_analysis.py: a module that contains helpful functions for use in data_analysis. Specifically, the functions pertain to maximum likelihood estimation and CGW analysis.

- generate_data.py: a module that contains a function to generate simulated data. Data can be simulated in 1, 2, or 3 dimensions.

- io.py: a module that contains the necessary functions to load test data.

- model.py: a module that contains the diffusion class used in this package, as well as the log_prior, log_likelihood, and log_posterior functions.

- printable.py: a module that contains the printable class, used in defining prior objects in prior.py.

- prior.py: a module that contains prior classes for jeffreys and uniform priors.

## In Progress
- finding a way to work around computational limits on evaluating the likelihood function in marginalization

- accounting for particle motion beyond purely diffusive motion in the model

- converting the module to use pandas instead of the custom 'diffusion' class