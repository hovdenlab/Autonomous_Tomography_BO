from skopt import Optimizer, dump, load
from skopt.space.space import Integer, Categorical, Real
from skopt.plots import plot_gaussian_process, plot_convergence, plot_objective_2D, plot_objective
from sklearn.gaussian_process.kernels import RBF, Matern
from skopt.learning import GaussianProcessRegressor as GPR
import random
import numpy as np
from datetime import datetime

# BO Parameters
nBOiter = 15
lower_bound = 1
upper_bound = 5000
n_init_pts = 3
# nu controls the smoothness of the matern kernel fitting for GP (e.g. nu = 1.5  corresponds to once differentiable functions, and nu = 2.5 to twice differentiable functions).
nu = 2.5 
# Prepare reconstruction script:
reconstructor_object = simulation_utils.FISTA_simulation()
# Prepare optimizer:
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
# Main Loop
for i in range(nBOiter):
    # Ask optimizer for next parameter to evaluate
    next_lambda = opt.ask()
    # Perform reconstruction, evaluate the performance (this function call can be replaced for any general black box function of interest)
    [data_error, recon] = reconstructor_object.FISTA_recon(next_lambda)
    # Tell the optimizer the result of the reconstruction
    res = opt.tell(next_lambda, data_error)
    # Save all data of interest (these are placeholder functions and should be replaced with functions that only save data you care about):
    save_results(data_error, recon, i+1)
    save_plot(plot_gaussian_process(res))
