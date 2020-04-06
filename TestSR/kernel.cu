
#include "Kernel.h"

#include <stdio.h>
#include <math.h>


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda()
{
	const int size = 10;
	int c[size];
	int a[size];
	int b[size],

    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    //cudaStatus = cudaSetDevice(0);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    //    goto Error;
    //}

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

//--------------------------------------------------------------------------
__device__ void quicksort(float *number, int first, int last) {
	int i, j, pivot;
	float temp;

	if (first < last) {
		pivot = first;
		i = first;
		j = last;

		while (i < j) {
			while (number[i] <= number[pivot] && i < last)
				i++;
			while (number[j] > number[pivot])
				j--;
			if (i < j) {
				temp = number[i];
				number[i] = number[j];
				number[j] = temp;
			}
		}

		temp = number[pivot];
		number[pivot] = number[j];
		number[j] = temp;
		quicksort(number, first, j - 1);
		quicksort(number, j + 1, last);

	}
}

//-------------------------------------
__device__ void BubbleSort_(float *pData, int size)
{
	int i, j;
	bool swapped;
	for (i = 0; i < size - 1; i++)
	{
		swapped = false;
		for (j = 0; j < size - i - 1; j++)
		{
			if (pData[j] > pData[j + 1])
			{
				float tmp = pData[j];
				pData[j] = pData[j + 1];
				pData[j + 1] = tmp;
				swapped = true;
			}
		}

		// IF no two elements were swapped by inner loop, then break 
		if (swapped == false)
			break;
	}
}
//---------------------------------------------------------------
__device__ void NormalSort_(float *pData, int size)
{
	for (int i = 0; i < size; i++) {
		float fCur = pData[i];
		for (int j = i + 1; j < size; j++) {
			if (fCur > pData[j]) {
				//Swap the variables.
				float tmp = fCur;
				fCur = pData[j];
				pData[i] = fCur;
				pData[j] = tmp;
			}
		}
	}
}

//-------------------------------------------------------------------
__global__ void medianFilterKernel(float *inputImageKernel, float *outputImagekernel, int imageWidth, int imageHeight, int dir)

{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float filterVector[40];// = { 0 };   //Take fiter window

	unsigned int iOffset = row * imageWidth + col;
	int iDirStep = dir == 0 ? 1 : imageWidth;
	if ((row == 0) || (col == 0) || (row == imageHeight - 1) || (col == imageWidth - 1))
	{
		outputImagekernel[iOffset] = 0; //Deal with boundry conditions
	}
	else 
	{

		for (int x = 0; x < 40; x++) 
		{
			float fVal =  inputImageKernel[iOffset + iDirStep*x];   // setup the filterign window.

			filterVector[x] = fVal;
		}
		
		//quicksort(filterVector, 0, 39);
		NormalSort_(filterVector, 40);
		//BubbleSort_(filterVector, 40);

		
		outputImagekernel[iOffset] = filterVector[20];   //Set the output variables.
	}
	//__syncthreads();
}


//-------------------------------------------------------------------
__global__ void medianFilterKernel2(float *inputImageKernel, float *outputImagekernel, int imageWidth, int imageHeight, int dir)

{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float filterVector[16*16][40];// = { 0 };   //Take fiter window

	unsigned int iOffset = row * imageWidth + col;
	int iDirStep = dir == 0 ? 1 : imageWidth;
	if ((row == 0) || (col == 0) || (row == imageHeight - 1) || (col == imageWidth - 1))
	{
		outputImagekernel[iOffset] = 0; //Deal with boundry conditions
	}
	else
	{

		for (int x = 0; x < 40; x++)
		{
			float fVal = inputImageKernel[iOffset + iDirStep * x];   // setup the filterign window.

			filterVector[threadIdx.y*16 + threadIdx.x][x] = fVal;
		}
		__syncthreads();
		//quicksort(filterVector, 0, 39);
		NormalSort_(filterVector[threadIdx.y * 16 + threadIdx.x], 40);
		//BubbleSort_(filterVector, 40);


		outputImagekernel[iOffset] = filterVector[threadIdx.y * 16 + threadIdx.x][20];   //Set the output variables.
	}
	//__syncthreads();
}




//--------------------------------------------------------------------------
cudaError_t SetCudaDevice(int id)
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(id);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		
	}
	return cudaStatus;
}

//--------------------------------------------------------------------------------
cudaError_t HorizontalMedianFilterCuda(float *pInData, float *pOutData, int iWidth, int iHeight, int iWin, int dir)
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
	medianFilterKernel <<<dimGrid, dimBlock>>>(d_InData, d_OutData, iWidth, iHeight, dir);

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


