#include "Matrix3D.h"
#include "matrix_ops.h"
#include <Eigen/Core>

using namespace Eigen;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

Matrix3D::Matrix3D()
{
}

Matrix3D::Matrix3D(int Nx, int Ny, int Nz)
{
	nx = Nx;
	ny = Ny;
	nz = Nz;
	data = new float [nx*ny*nz];
	size = nx * ny * nz;
}

// for (int i=0; i<nx; i++) {
    // for (int j=0; j<ny; j++) {
        // for (int k=0; k<nz; k++)
    // }
// }

// 
void Matrix3D::setData(Mat inBuffer, int slice)
{
    for (int yInd = 0; yInd < ny; yInd++) {
      for (int zInd = 0; zInd < nz; zInd++) {
          data[index(slice,yInd,zInd)] = inBuffer(yInd,zInd);
      }
    }
}

// Return Reconstruction to Python.
Mat Matrix3D::getData(int slice)
{
    Mat outBuffer(ny,nz);
    for (int yInd = 0; yInd < ny; yInd++) {
        for (int zInd = 0; zInd < nz; zInd++) {
            outBuffer(yInd,zInd) = data[index(slice,yInd,zInd)];
        }
    }
    return outBuffer;
}

// Get Single Value from (x,y,z) input
float Matrix3D::get_val(int i,int j,int k) { return data[(ny*nz)*i + nz*j + k]; }

// Get Index from (x,y,z) input
int Matrix3D::index(int i, int j, int k){ return (ny*nz)*i + nz*j + k; } 
//    return i + nx*j + (nx*ny)*k;

// Set the Background to Single Value
void Matrix3D::setBackground(int backgroundValue) { cuda_set_background(data,backgroundValue,nx,ny,nz); }

// Sum All the Values in Reconstruction 
float Matrix3D::sum() { return cuda_sum(data,nx,ny,nz); }

// L1 Norm
float Matrix3D::l1_norm() { return cuda_l1_norm(data,nx,ny,nz); }

// L2 Norm 
float Matrix3D::norm() { return cuda_norm(data,nx,ny,nz); }

// Apply Positivity to Reconstruction 
void Matrix3D::positivity() { cuda_positivity(data,nx,ny,nz); }

// Soft Thresholding Operation
void Matrix3D::soft_threshold(float lambda) { cuda_soft_threshold(data,lambda,nx,ny,nz); }