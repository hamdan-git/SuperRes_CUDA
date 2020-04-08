#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_texture_types.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <vector>

cudaError_t addWithCuda();

cudaError_t HorizontalMedianFilterCuda(float *pInData, float *pOutData, int iWidth, int iHeight, int iWin, int dir);
cudaError_t SetCudaDevice(int id);


cudaError_t interpolateBetweenColumnsCuda(float *pInData, float *pOutData, int iWidth, int iHeight, int startX, int endX);
cudaError_t Filtering_1D_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, float *pKernel, int iKernelSize, int dir);
cudaError_t Filtering_1D_shared_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, float *pKernel, int iKernelSize, int dir);
cudaError_t Filtering_1D_tex_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, float *pKernel, int iKernelSize, int dir);
cudaError_t SimWin_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, unsigned char *pMapData);
cudaError_t SimWin_tex_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, int iNumFrames, unsigned char *pMapData);
cudaError_t RotateImage_tex_Cuda(float* pInData, int inWidth, int inHeight, float *pOutData, int outWidth, int outHeight, double theta);
//cudaError_t InterpolateBetweenFilterPoints_Cuda(float *pInData, float *pOutData, float *pRef1, float *pRef2, float *pGain1, float *pGain2, int iWidth, int iHeight, float fOneOver_s, int iRefPointIndex, int iNumPoints);
//cudaError_t InterpolateBetweenFilterPoints_Cuda2(float *pInData, float *pOutData, float *pRef1, float *pRef2, int iWidth, int iHeight, float fOneOver_s, int iRefPointIndex, int iNumPoints);
cudaError_t InterpolateBetweenFilterPoints_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, int iNumFrames, std::vector<float*> *pGain, int iNumPoints, int iNumBins, float fps);
cudaError_t RotateAddImage_tex_Cuda(unsigned short* pInData, int inWidth, int inHeight, int iNumFrames, unsigned short *pOutData, int outWidth, int outHeight, double theta, double fScale, double fMag);

