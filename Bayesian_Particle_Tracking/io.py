from Bayesian_Particle_Tracking import model
import numpy as np
import os

#T=300, mu = 10^-4, a = 10^-8
#=> D = 2.19*10^-10
# TODO: FIXME: You generally don't want to be running actual computations in files that are imported. Move this data generation to a seperate script you run explicitly
data = model.generator(1000,10**(-8),10**(-4),10**(-8),[0,0,0], T=300)
np.save('test_data', data)

def get_example_data_file_path(filename, data_dir='example_data'):
    # __file__ is the location of the source file currently in use (so
    # in this case io.py). We can use it as base path to construct
    # other paths from that should end up correct on other machines or
    # when the package is installed
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    # If you need to go up another directory (for example if you have
    # this function in your tests directory and your data is in the
    # package directory one level up) you can use
    # up_dir = os.path.split(start_dir)[0]
    data_dir = os.path.join(start_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)

new_data = np.load(get_example_data_file_path('test_data.npy', ''))

input_data = model.diffusion(new_data)
