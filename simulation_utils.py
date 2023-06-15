import gpu_3D.Utils.astra_ctvlib as astra_ctvlib
import gpu_3D.Utils.pytvlib as pytvlib
from tqdm import tqdm
import numpy as np
import h5py

class FISTA_simulation:
    def __init__(self):
        # Load the Original Volume
        file = h5py.File(<your_file>,'r')  # Specify your input dataset
        self.original_volume = file['vol'][:]
        file.close()
        # Read Dimensions of Test Object
        (self.Nx, self.Ny, self.Nz) = self.original_volume.shape
        # Set your tomography collection parameters, e.g. ±70º with a +2º tilt increment:
        tiltAngles = np.arange(-70,70,2)  # Modify this per experiment
        # Initialize the C++ Object..
        self.tomography_object = astra_ctvlib.astra_ctvlib(self.Nx, self.Ny, np.deg2rad(tiltAngles))
        ## astra_ctvlib by default creates one 3D volume for the reconstruction, any additional volumes needs to be externally intialized (this is to save memory consumption) ##
        self.tomography_object.initialize_initial_volume()
        # Optional: Apply Poisson Noise to Background Volume
        self.original_volume[self.original_volume == 0] = 1
        # Let's pass the volume from python to C++  
        for s in range(Nx):
            self.tomography_object.set_original_volume(self.original_volume[s,:,:],s)
        # Now Let's Create the Projection Images
        self.tomography_object.create_projections()
        # Optional: Apply poisson noise to volume.
        SNR = 5     # Modify this to simulate different experimental conditions
        if SNR != 0: self.tomography_object.poisson_noise(SNR)

    ### Perform FISTA reconstruction ###
    def FISTA_recon(self, lambdaParam):
        self.tomography_object.restart_recon()
        # Inialize the Reconstruction Algorithm
        alg = 'fista'; prox = 'tv'
        pytvlib.initialize_algorithm(self.tomography_object,alg)
        # Reconstruction Parameters
        Niter = 50; nTViter = 10
        # Momentum and Objective Function 
        fista_cost = np.zeros(Niter); t0 = 1
        # Main Loop 
        for k in tqdm(range(Niter)):
            # Gradient Step
            pytvlib.run(self.tomography_object,alg)    
            # Threshold Step
            if prox == 'tv': tv = self.tomography_object.tv_fgp(nTViter,lambdaParam)
            else:            self.tomography_object.soft_threshold(lambdaParam) 
            # Momentum Step
            tk = 0.5 * (1 + np.sqrt(1 + 4 * t0**2))
            self.tomography_object.fista_momentum((t0-1)/tk)
            t0 = tk
            # Measure Objective  
            if prox == 'tv': fista_cost[k] = 0.5 * self.tomography_object.data_distance()**2 + lambdaParam * self.tomography_object.tv()
            else:            fista_cost[k] = 0.5 * self.tomography_object.data_distance()**2 + lambdaParam * self.tomography_object.l1_norm()
        # Return the Reconstruction to Python
        recon = np.zeros([self.Nx, self.Ny, self.Nz],dtype=np.float32)
        for s in range(self.Nx):
            recon[s,] = self.tomography_object.get_recon(s)
        return [self.tomography_object.rmse(), recon]
