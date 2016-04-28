from Bayesian_Particle_Tracking import model, generate_data
import numpy as np
import os

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
    up_dir = os.path.split(start_dir)[0]
    data_dir = os.path.join(up_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)

def get_example_model(filename):
    new_data = np.load(get_example_data_file_path('test_data.npy', ''))
    return model.diffusion(new_data)

#TODO: Evaluate whether or not two separate data sets are actually necessary
#compare_data = np.load(get_example_data_file_path('compare_data.npy', ''))
#compare_input = model.diffusion(new_data)
