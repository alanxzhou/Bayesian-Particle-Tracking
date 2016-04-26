# Bayesian-Particle-Tracking

Background: Accurate estimation of the diffusion coefficient and its uncertainty is important in adequately understanding any diffusion process. From a calculation of the diffusion coefficient, one can directly determine properties of the particle (size) or of the medium (dynamic viscosity). There does not seem to exist a standard accepted procedure for calculating the diffusion coefficient. Current methods are very ad hoc and can be improved upon. 

With some relatively simple Bayesian statistics, the diffusion coefficient, and its uncertainty can be accurately measured. This package is designed to allow bayesian analysis of particle tracking data. Specifically, it allows estimations of the diffusion coefficient for a diffusion process in 3D particle tracking given a trajectory and the uncertainty on the measurments in that trajectory.

Parameters in determining trajectory of a 3D Diffusion Process:

sigma: uncertainty in measurement on position

mu: dynamic viscosity of the medium

a: radius of the particle

T: temperature

D: diffusion coefficient; this is a function of mu, a, and T. Specifically by Stokes-Einsten:
D = kb*T/(6*pi*mu*a)

Data: The package tutorial uses simulated data to demonstrate its efficacy. This is because it is much easier to demonstrate efficacy when the desired answer is known.

Future plans:
Develop a model that has dependencies on the parameters of the raw data (particle tracking images)