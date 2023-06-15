# Source Code - Autonomous Electron Tomography Reconstruction with Machine Learning
**Generating superior reconstructions using CS algorithms and Bayesian optimization**

William Millsaps<sup>1</sup>, Jonathan Schwartz<sup>2†</sup>, Zichao Wendy Di<sup>3</sup>, Yi Jiang<sup>4</sup>, Robert Hovden<sup>2,5††</sup>

*<sup>1</sup>Department of Nuclear Engineering and Radiological Sciences, University of Michigan, Ann Arbor, MI.* 
*<sup>2</sup>Department of Materials Science and Engineering, University of Michigan, Ann Arbor, MI.*    
*<sup>3</sup>Mathematics and Computer Science Division, Argonne National Laboratory, Lemont, IL.*     
*<sup>4</sup>Advanced Photon Source Facility, Argonne National Laboratory, Lemont, IL.*     
*<sup>5</sup>Applied Physics Program, University of Michigan, Ann Arbor, MI.*
*Correspondence and requests for materials should be addressed to J.S.† (jtschw@umich.edu) and to R.H.†† (hovden@umich.edu).* 

[If you use any of the data and source codes in your publications and/or presentations, we request that you cite our paper: W. Millsaps, J. Schwartz, et. al., "Autonomous Electron Tomography Reconstruction with Machine Learning", _Microscopy & Microanalysis._(2023).](Manuscript ID MAM-23-045.R1, link soon)

# Installation 

To clone the repositiory run: 

` git clone --recursive https://github.com/hovdenlab/Autonomous_Tomography_BO.git`

Install the astra-toolbox (https://www.astra-toolbox.com/). In the make.inc file in utils specify the location of the astra-toolbox library on your local machine. Compile the regularization, container, and astra scripts using the three Makefiles (in utils, utils/regularizers, and utils/container) prior to running the example jupyter notebook or example python file.

The bayesian_tomography.py and bayesian_tomography_notebook.ipynb are two versions of the same file that reproduce the results of the manuscript, be sure to specify your input filename, and modify the input file type as necessary in the simulation_utils_FISTA initialization. To port additional reconstruction algorithms or functions of interest, ignore simulation_utils_FISTA and edit line 23 in bayesian_tomography.py (where the reconstruction_object is created) to implement BO for additional applications.
