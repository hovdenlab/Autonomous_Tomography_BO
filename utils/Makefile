#Makefile
#!/bin/sh
include ./make.inc

GPUCONFIG = astra_ctvlib`python3-config --extension-suffix`

all: shared_library astra_ctvlib

shared_library: 
	nvcc -shared container/*.o regularizers/*.o -o aux_func.so

astra_ctvlib: astra_ctvlib.cpp astra_ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) astra_ctvlib.cpp -o $(GPUCONFIG) $(ASTRA_LIB) aux_func.so

clean:
	rm -rf *.so *.o
