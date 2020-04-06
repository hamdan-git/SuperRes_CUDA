
#include "Kernel.h"

#include <stdio.h>
#include <math.h>
#include <float.h>


//---------------------------------------------------------------------------------------------
__device__ int GetYPosGivenSlope_MCP_(int iRefX, float fSlope, int iCurX, int iCurY)
{
	return (int)(fSlope*(float)(iRefX - iCurX)) + iCurY;
}

//-----------------------------------------------------------------------------------------------
__device__ float GetDistanceBetweenTwoPoints_MCP_(int x1, int y1, int x2, int y2)
{
	return (float)sqrt((double)((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)));
}

//----------------------------------------------------------------------------------
__device__ float InterpolatePixelFrom2Points_MCP_(float fVal1, float fDist1, float fVal2, float fDist2)
{
	float fDenom = fDist1 + fDist2;
	if (fDenom != 0.0f)
	{
		fDenom = 1.0f / fDenom;
		return (fVal1*fDist2 + fVal2 * fDist1)*fDenom;
	}
	return 0.0;
}



//-------------------------------------------------------------------
__global__ void interpolateBetweenColumnsKernel(float *inputImageKernel, float *outputImagekernel, int imageWidth, int imageHeight, int startX, int endX)

{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int iOffset = row * imageWidth + col;	

	if (col < startX || col > endX)
	{
		outputImagekernel[iOffset] = inputImageKernel[iOffset];
		return;
	}

	float fCureVal = inputImageKernel[iOffset];
	//go to different directions
	float fMinGrad = FLT_MAX;
	int iMinIndex = -1;
	float fSlopes[3] = { -1.0f, 0.0f, 1.0f }; //3 directions to search for best fit
	float fVals[3]; //containg the interpolated values in different directions
	float fGrad[3]; //containing gradient in different directions
	for (int k = 0; k < 3; k++)
	{
		int iY1 = GetYPosGivenSlope_MCP_(startX, fSlopes[k], col, row);
		//iY1 = max(min(iY1, imageHeight - 1), 0);
		iY1 = iY1 > imageHeight - 1 ? imageHeight - 1 : iY1;
		iY1 = iY1 < 0 ? 0 : iY1;
		int iY2 = GetYPosGivenSlope_MCP_(endX, fSlopes[k], col, row);
		//iY2 = max(min(iY2, imageHeight - 1), 0);
		iY2 = iY2 > imageHeight - 1 ? imageHeight - 1 : iY2;
		iY2 = iY2 < 0 ? 0 : iY2;
		float fDist1 = GetDistanceBetweenTwoPoints_MCP_(startX, iY1, col, row);
		float fDist2 = GetDistanceBetweenTwoPoints_MCP_(endX, iY2, col, row);
		//get the pixel values
		float fVal1 = inputImageKernel[iY1*imageWidth + startX];
		float fVal2 = inputImageKernel[iY2*imageWidth + endX];

		//find the interpolated values between the two pixels using distance as a contributing factor
		fVals[k] = InterpolatePixelFrom2Points_MCP_(fVal1, fDist1, fVal2, fDist2);
		float fCurGrad = (float)fabs(fVal1 - fVal2);
		fGrad[k] = fCurGrad;
		if (fCurGrad < fMinGrad)
		{
			fMinGrad = fCurGrad; iMinIndex = k;
		}
		//iMinIndex = 1;
	}

	if (iMinIndex >= 0)
	{
		outputImagekernel[iOffset] = fVals[iMinIndex];
	}

	//__syncthreads();
}



//--------------------------------------------------------------------------------
cudaError_t interpolateBetweenColumnsCuda(float *pInData, float *pOutData, int iWidth, int iHeight, int startX, int endX)
{

	float *d_InData = 0;
	float *d_OutData = 0;

	int iFrameSize = iWidth * iHeight;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	//cudaStatus = cudaSetDevice(0);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	//	goto Error;
	//}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&d_InData, iFrameSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_OutData, iFrameSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(d_InData, pInData, iFrameSize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int TILE_SIZE_X = 16;
	int TILE_SIZE_Y = 16;
	dim3 dimBlock(TILE_SIZE_X, TILE_SIZE_Y);

	dim3 dimGrid((int)ceil((float)iWidth / (float)TILE_SIZE_X), (int)ceil((float)iHeight / (float)TILE_SIZE_Y));

	// Launch a kernel on the GPU with one thread for each element.
	interpolateBetweenColumnsKernel << <dimGrid, dimBlock >> > (d_InData, d_OutData, iWidth, iHeight, startX, endX);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(pOutData, d_OutData, iFrameSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_InData);
	cudaFree(d_OutData);

	return cudaStatus;
}


