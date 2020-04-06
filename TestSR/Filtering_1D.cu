
#include "Kernel.h"

#include <stdio.h>
#include <math.h>
#include <float.h>


//----------------------------------------------------------------------------------------------------------
__device__ void Filtering_1D_vertical_dev(float *inputImageKernel, float *outputImagekernel, int imageWidth, int imageHeight, float *pKernel, int iKernelSize, int row, int col)
{
	unsigned int iOffset = row * imageWidth + col;
	float sum = 0;
	int iWin = iKernelSize / 2;

	if (row < iWin)
	{
		for (int k = 0; k <= iWin; k++)
		{
			sum += inputImageKernel[iOffset + (k + row) * imageWidth] * pKernel[k + iWin];  //reverese of data with the rest of the kernel
		}
		for (int k = 1; k <= iWin; k++)
		{
			sum += inputImageKernel[iOffset + abs(k - row) * imageWidth] * pKernel[iWin - k];
		}
	}
	else if (row >= imageHeight - iWin)
	{
		for (int k = 0; k <= iWin; k++)
		{
			sum += inputImageKernel[iOffset - ((k + (imageHeight - row)) * imageWidth)] * pKernel[k + iWin];  //reverese of data with the rest of the kernel
		}
		for (int k = 1; k <= iWin; k++)
		{
			sum += inputImageKernel[iOffset - (abs(k - (imageHeight - row)) * imageWidth)] * pKernel[iWin - k];
		}
	}
	else
	{
		for (int k = -iWin; k <= iWin; k++)
		{
			int iNewRow = row + k;
			if (iNewRow >= 0 && iNewRow < imageHeight)
				sum += inputImageKernel[iOffset + k * imageWidth] * pKernel[k + iWin];
		}
	}

	outputImagekernel[iOffset] = sum;
}

//----------------------------------------------------------------------------------------------------------
__device__ void Filtering_1D_horizontal_dev(float *inputImageKernel, float *outputImagekernel, int imageWidth, int imageHeight, float *pKernel, int iKernelSize, int row, int col)
{
	unsigned int iOffset = row * imageWidth + col;
	float sum = 0;
	int iWin = iKernelSize / 2;

	if (col < iWin)
	{
		for (int k = 0; k <= iWin; k++)
		{
			sum += inputImageKernel[iOffset + (k + col)] * pKernel[k + iWin];  //reverese of data with the rest of the kernel
		}
		for (int k = 1; k <= iWin; k++)
		{
			sum += inputImageKernel[iOffset + abs(k - col)] * pKernel[iWin - k];
		}
	}
	else if (col >= imageWidth - iWin)
	{
		for (int k = 0; k <= iWin; k++)
		{
			sum += inputImageKernel[iOffset - ((k + (imageWidth - col)) )] * pKernel[k + iWin];  //reverese of data with the rest of the kernel
		}
		for (int k = 1; k <= iWin; k++)
		{
			sum += inputImageKernel[iOffset - (abs(k - (imageWidth - col)) )] * pKernel[iWin - k];
		}
	}
	else
	{
		for (int k = -iWin; k <= iWin; k++)
		{
			int iNewCol = col + k;
			if (iNewCol >= 0 && iNewCol < imageWidth)
				sum += inputImageKernel[iOffset + k ] * pKernel[k + iWin];
		}
	}

	outputImagekernel[iOffset] = sum;
}

//-------------------------------------------------------------------
__global__ void Filtering_1D_Kernel(float *inputImageKernel, float *outputImagekernel, int imageWidth, int imageHeight, float *pKernel, int iKernelSize, int dir)

{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;



	if (dir == 0) //horizontal only
	{
		Filtering_1D_horizontal_dev(inputImageKernel, outputImagekernel, imageWidth, imageHeight, pKernel, iKernelSize, row, col);

	}
	else
	{
		Filtering_1D_vertical_dev(inputImageKernel, outputImagekernel, imageWidth, imageHeight, pKernel, iKernelSize, row, col);
	}


}



//--------------------------------------------------------------------------------
cudaError_t Filtering_1D_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, float *pKernel, int iKernelSize, int dir)
{

	float *d_InData = 0;
	float *d_OutData = 0;
	float *d_pKernel = 0;

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

	cudaStatus = cudaMalloc((void**)&d_pKernel, iKernelSize * sizeof(float));
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

	//copy kernel
	cudaStatus = cudaMemcpy(d_pKernel, pKernel, iKernelSize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int TILE_SIZE_X = 16;
	int TILE_SIZE_Y = 16;
	dim3 dimBlock(TILE_SIZE_X, TILE_SIZE_Y);

	dim3 dimGrid((int)ceil((float)iWidth / (float)TILE_SIZE_X), (int)ceil((float)iHeight / (float)TILE_SIZE_Y));


	// Launch a kernel on the GPU with one thread for each element.
		Filtering_1D_Kernel << <dimGrid, dimBlock >> > (d_InData, d_OutData, iWidth, iHeight, d_pKernel, iKernelSize, dir);

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
	cudaFree(d_pKernel);

	return cudaStatus;
}


