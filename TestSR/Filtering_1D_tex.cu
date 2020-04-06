
#include "Kernel.h"

#include <stdio.h>
#include <math.h>
#include <float.h>


// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex_filt1D_img;


//----------------------------------------------------------------------------------------------------------
__device__ void Filtering_1D_tex_vertical_dev( float *outputImagekernel, int imageWidth, int imageHeight, float *pKernel, int iKernelSize, int row, int col)
{
	unsigned int iOffset = row * imageWidth + col;
	float sum = 0;
	int iWin = iKernelSize / 2;

	if (row < iWin)
	{
		for (int k = 0; k <= iWin; k++)
		{
			sum += tex2D(tex_filt1D_img, col, row+k) * pKernel[k + iWin];  //reverese of data with the rest of the kernel
		}
		for (int k = 1; k <= iWin; k++)
		{
			sum += tex2D(tex_filt1D_img, col, abs(k-row)) * pKernel[iWin - k];
		}
	}
	else if (row >= imageHeight - iWin)
	{
		for (int k = 0; k <= iWin; k++)
		{
			sum += tex2D(tex_filt1D_img, col, k + (imageHeight - row)) * pKernel[k + iWin];  //reverese of data with the rest of the kernel
		}
		for (int k = 1; k <= iWin; k++)
		{
			sum += tex2D(tex_filt1D_img, col, abs(k - (imageHeight - row))) * pKernel[iWin - k];
		}
	}
	else
	{
		for (int k = -iWin; k <= iWin; k++)
		{
			int iNewRow = row + k;
			if (iNewRow >= 0 && iNewRow < imageHeight)
			{
				sum += tex2D(tex_filt1D_img, col, iNewRow) * pKernel[k + iWin];
			}
		}
	}

	outputImagekernel[iOffset] = sum;
}

//----------------------------------------------------------------------------------------------------------
__device__ void Filtering_1D_tex_horizontal_dev(float *outputImagekernel, int imageWidth, int imageHeight, float *pKernel, int iKernelSize, int row, int col)
{
	unsigned int iOffset = row * imageWidth + col;
	float sum = 0;
	int iWin = iKernelSize / 2;

	if (col < iWin)
	{
		for (int k = 0; k <= iWin; k++)
		{
			sum += tex2D(tex_filt1D_img, k+col, row) * pKernel[k + iWin];  //reverese of data with the rest of the kernel
		}
		for (int k = 1; k <= iWin; k++)
		{
			sum += tex2D(tex_filt1D_img, abs(k - col), row) * pKernel[iWin - k];
		}
	}
	else if (col >= imageWidth - iWin)
	{
		for (int k = 0; k <= iWin; k++)
		{
			sum += tex2D(tex_filt1D_img, k + (imageWidth - col), row) * pKernel[k + iWin];  //reverese of data with the rest of the kernel
		}
		for (int k = 1; k <= iWin; k++)
		{
			sum += tex2D(tex_filt1D_img,  abs(k - (imageWidth - col)), row) * pKernel[iWin - k];
		}
	}
	else
	{
		for (int k = -iWin; k <= iWin; k++)
		{
			int iNewCol = col + k;
			if (iNewCol >= 0 && iNewCol < imageWidth)
			{
				sum += tex2D(tex_filt1D_img,  k+col, row) * pKernel[k + iWin];
			}
		}
	}

	outputImagekernel[iOffset] = sum;
}

//-------------------------------------------------------------------
__global__ void Filtering_1D_tex_Kernel( float *outputImagekernel, int imageWidth, int imageHeight, float *pKernel_g, int iKernelSize, int dir)

{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float pKernel[111]; //up to win=5
	for (int i = 0; i < iKernelSize; i++)
		pKernel[i] = pKernel_g[i];

	__syncthreads(); //make sure all threads have loaded the kernel in the shared mem

	if (dir == 0) //horizontal only
	{
		Filtering_1D_tex_horizontal_dev( outputImagekernel, imageWidth, imageHeight, pKernel, iKernelSize, row, col);

	}
	else
	{
		Filtering_1D_tex_vertical_dev( outputImagekernel, imageWidth, imageHeight, pKernel, iKernelSize, row, col);
	}


}



//--------------------------------------------------------------------------------
cudaError_t Filtering_1D_tex_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, float *pKernel, int iKernelSize, int dir)
{

	cudaArray *cuArray_img;
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
	//imput image text array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMallocArray(&cuArray_img, &channelDesc, iWidth, iHeight);
	cudaMemcpyToArray(cuArray_img, 0, 0, pInData, iFrameSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(tex_filt1D_img, cuArray_img, channelDesc);

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
	Filtering_1D_tex_Kernel << <dimGrid, dimBlock >> > (d_OutData, iWidth, iHeight, d_pKernel, iKernelSize, dir);

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
	cudaFreeArray(cuArray_img);
	cudaFree(d_OutData);
	cudaFree(d_pKernel);

	return cudaStatus;
}


