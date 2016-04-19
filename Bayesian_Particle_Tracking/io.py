from Bayesian_Particle_Tracking import model
import numpy as np

#T=300, mu = 10^-4, a = 10^-8
#=> D = 2.19*10^-10
data = model.generator(1000,10**(-8),10**(-4),10**(-8),[0,0,0], T=300)
np.save('test_data', data)

#This section is specific to my computer: to be removed.
home_dir = "/Users/alanzhou/Documents/Physics_201/final_project/Bayesian-Particle-Tracking/Bayesian_Particle_Tracking/"
new_data = np.load(home_dir + 'test_data.npy')

input_data = model.diffusion(new_data)