import astra_ctvlib
import pytvlib
from tqdm import tqdm
import numpy as np
import h5py

class FISTA_simulation:

    def __init__(self, fname,tiltAngles, SNR=5):

        # Load the Original Volume (customize this per input file type)
        file = h5py.File(fname,'r')
        self.original_volume = file['recon'][:]
        file.close()

        # Read Dimensions of Test Object
        (self.Nx, self.Ny, self.Nz) = self.original_volume.shape

        # Initialize the C++ Object..
        self.tomography_object = astra_ctvlib.astra_ctvlib(self.Nx, self.Ny, np.deg2rad(tiltAngles))

        ## astra_ctvlib by default creates one 3D volume for the reconstruction, any additional volumes needs to be externally intialized (this is to save memory consumption) ##
        self.tomography_object.initialize_initial_volume()

        # Optional: Apply Poisson Noise to Background Volume
        self.original_volume[self.original_volume == 0] = 1

        # Let's pass the volume from python to C++
        for s in range(self.Nx):
            self.tomography_object.set_original_volume(self.original_volume[s,:,:],s)

        # Now Let's Create the Projection Images
        self.tomography_object.create_projections()

        # Optional: Apply poisson noise to volume
        if SNR != 0: self.tomography_object.poisson_noise(SNR)

    ### Perform FISTA reconstruction ###
    def FISTA_recon(self, params):

        # Initialize Reconstruction
        self.tomography_object.restart_recon()
        self.rmse_vec, self.dd_vec, self.tv_vec = np.zeros(params['Niter']), np.zeros(params['Niter']), np.zeros(params['Niter'])

        # Inialize the Reconstruction Algorithm
        alg = params['alg']
        pytvlib.initialize_algorithm(self.tomography_object,alg)

        # Reconstruction Parameters
        self.params = params
        lambdaParam = params['lambdaParam']; Niter = params['Niter']; nTViter = params['nTViter']

        # Momentum and Objective Function 
        self.fista_cost = np.zeros(Niter); t0 = 1

        ### Main Loop ### 
        for k in tqdm(range(Niter)):

            # Gradient Step
            pytvlib.run(self.tomography_object,alg)    

            # Threshold Step
            self.tomography_object.tv_fgp(nTViter,lambdaParam)

            # Momentum Step
            tk = 0.5 * (1 + np.sqrt(1 + 4 * t0**2))
            self.tomography_object.fista_momentum((t0-1)/tk)
            t0 = tk

            # Measure Objective  
            self.fista_cost[k] = 0.5 * self.tomography_object.data_distance()**2 + lambdaParam * self.tomography_object.tv()

            # Measure other performance metrics
            self.dd_vec[k] = self.tomography_object.data_distance()
            self.tv_vec[k] = self.tomography_object.tv()
            self.rmse_vec[k] = self.tomography_object.rmse()

        # Return the Reconstruction to Python
        self.recon = np.zeros([self.Nx, self.Ny, self.Nz],dtype=np.float32)
        for s in range(self.Nx):
            self.recon[s,] = self.tomography_object.get_recon(s)


        return [self.tomography_object.rmse()]
    
    def save_results(self,fname, groupName, i):
        h5File = h5py.File(fname,'a')
        group = h5File.create_group(groupName+'/'+str(i))
        group.create_dataset('LambdaParam', data=self.params['lambdaParam'])
        group.create_dataset('RMSE', data= self.rmse_vec)
        group.create_dataset('Reconstruction', data=self.recon[140,:,:])  #Customize to save reconstruction slice
        group.create_dataset('DD',data=self.dd_vec)
        group.create_dataset('TV',data=self.tv_vec)
        group.create_dataset('FISTA_cost',data=self.fista_cost)
        h5File.close()
