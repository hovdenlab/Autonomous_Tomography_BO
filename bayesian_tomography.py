from skopt import Optimizer
from skopt.space.space import Real
from skopt.plots import plot_gaussian_process
from sklearn.gaussian_process.kernels import Matern
from skopt.learning import GaussianProcessRegressor as GPR
import simulation_utils_FISTA
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os

##### Prepare reconstruction script #####

# Test Phantom Object 
filename = 'your_file'

# Tilt Series for Experiment / Simulation
tiltAngles = np.arange(-70,70,2)
recon_parameters = {'lambdaParam':0, 'Niter': 100, 'nTViter':15, 'alg':'fista'}

# Class for Performing Reconstructions
reconstructor_object = simulation_utils_FISTA.FISTA_simulation(filename,tiltAngles)

##### Prepare optimizer #####

# BO Parameters
nBOiter = 20        # Total iterations (including initial pts)
lower_bound = 1
upper_bound = 5000
n_init_pts = 5
nu = 2.5            # nu controls the smoothness of the matern kernel fitting for GP (e.g. nu = 1.5 corresponds to once differentiable functions, and nu = 2.5 to twice differentiable functions).

# Define the Kernel
mat_kern = Matern(length_scale = 1, length_scale_bounds = (1, 10), nu = nu)
gpr = GPR(kernel = mat_kern, n_restarts_optimizer = 10)

# Define bounds for fit parameter
bounds = [Real(lower_bound, upper_bound, name = 'lambda')]
initial_sampling = np.linspace(lower_bound,  upper_bound, n_init_pts)
init_pts_generator = 'grid'

# Optimizer for BO
rng_seed = int(datetime.now().strftime('%f'))
random.seed(rng_seed)
opt = Optimizer(bounds, base_estimator = gpr, acq_func = 'gp_hedge',n_initial_points = n_init_pts, initial_point_generator=init_pts_generator, acq_optimizer='sampling',random_state=rng_seed)

# Parameters for plotting and saving Bayesian predictions
save_directory = 'Bayesian_Optimization_tomography/' # Your results folder
fname = 'lambdaParam_tuning.h5' # File with results
output_name = 'lambdaParam_explore' # h5 group name
if not os.path.exists(save_directory): os.makedirs(save_directory) # Make directory
kwargs = {'show_acq_func' :False, 'show_mu' :True, 'show_legend': False} # Optional arguments for plotting

# Main Loop
for i in range(nBOiter):

    # Ask optimizer for next parameter to evaluate
    next_lambda = opt.ask()[0]
    recon_parameters['lambdaParam'] = next_lambda

    # Perform reconstruction, evaluate the performance (this function call can be replaced for any general black box function of interest)
    data_error = reconstructor_object.FISTA_recon(recon_parameters)

    # Tell the optimizer the result of the reconstruction
    res = opt.tell([next_lambda], data_error)
    
    # Save data of interest:
    reconstructor_object.save_results(save_directory+fname, output_name, i+1)          # Save 2D Slices from the Reconstruction

    # Save plot of Bayesian Optimization
    if i >= n_init_pts:
        plt.figure()
        plot_gaussian_process(res,**kwargs)     # Recreate Parameter Estimate Plots from Fig. 3a
        plt.savefig(f'{save_directory}bo_gp{i+1}.png', transparent = False, bbox_inches = 'tight')
