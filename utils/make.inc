CXX = g++ -fPIC -fopenmp
MPXX = mpicxx -fPIC -fopenmp
CUDAXX = nvcc -shared -Xcompiler -fPIC -c
CXXFLAGS = -O3 -Wno-div-by-zero -shared -std=c++11  `python3 -m pybind11 --includes`
EIGEN = -I ../../thirdparty/eigen 
ASTRA = -DASTRA_CUDA -I /home/hlab/datadrive/tomography/astra-toolbox -I /home/hlab/datadrive/tomography/astra-toolbox/include
ASTRA_LIB = -L /home/hlab/datadrive/tomography/astra/lib/ -lastra
CUDA = -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lz
HDF5_INC = -I /home/hlab/tomography/tomo_TV/thirdparty/HDF5/include
HDF5_LIBS= -L /home/hlab/tomography/tomo_TV/thirdparty/HDF5/lib -lhdf5 -lhdf5_hl
NonPAR_HDF5_INC = -I /home/hlab/prismatic/HDF5/include
NonPAR_HDF5_LIBS = -L /home/hlab/prismatic/HDF5/lib -lhdf5 -lhdf5_hl
