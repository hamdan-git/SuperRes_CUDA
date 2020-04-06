#include "Kernel.h"

#include <stdio.h>
#include <math.h>
#include <float.h>


// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex_rot_img;

__device__ double getInterpolatedPixel(double x, double y, int iWidth, int iHeight, float* pixels)
{
	int xbase = (int)x;
	int ybase = (int)y;
	xbase = xbase < iWidth ? xbase : iWidth - 1;// min(xbase, iWidth - 1);
	ybase = ybase < iHeight ? ybase : iHeight - 1;// (ybase, iHeight - 1);
	//if (xbase  >= iWidth  ybase >= iHeight )
	//	return 1;
	double xFraction = x - xbase;
	double yFraction = y - ybase;
	int offset = ybase * iWidth + xbase;
	double lowerLeft = pixels[offset];
	double lowerRight = xbase == iWidth - 1 ? pixels[offset] : pixels[offset + 1];
	double upperRight = (xbase == iWidth - 1 || ybase == iHeight - 1) ? pixels[offset] : pixels[offset + iWidth + 1];
	double upperLeft = ybase == iHeight - 1 ? pixels[offset] : pixels[offset + iWidth];
	double upperAverage = upperLeft;
	if (xFraction != 0.0)
		upperAverage += xFraction * (upperRight - upperLeft);
	double lowerAverage = lowerLeft;
	if (xFraction != 0.0)
		lowerAverage += xFraction * (lowerRight - lowerLeft);
	if (yFraction == 0.0)
		return lowerAverage;
	else
		return lowerAverage + yFraction * (upperAverage - lowerAverage);
}

//-------------------------------------------------------------------
__global__ void RotateImage_tex_kernel(float *outputImagekernel, int inWidth, int inHeight, int outWidth, int outHeight, double theta)

{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float rads = (theta) * 3.1415926 / 180.0;

	float u = (float)col - (float)(outWidth) / 2.0f;
	float v = (float)row - (float)(outHeight) / 2.0f;
	float tu = u * cosf(rads) - v * sinf(rads);
	float tv = v * cosf(rads) + u * sinf(rads);

	//tu *= (float)outWidth / (float)inWidth;
	//tv *= (float)outHeight / (float)inHeight;
	tu /= (float)inWidth;
	tv /= (float)inHeight;



	if (/*tu < inWidth && tu>=0 && tv>=0 && tv < inHeight &&*/ col < outWidth && row < outHeight)
		outputImagekernel[row*outWidth + col] = tex2D(tex_rot_img, tu + 0.5f, tv + 0.5f);
	//outputImagekernel[row*outWidth + col] = tex2D(tex_rot_img, tu , tv);


}

__global__ void RotateImage_kernel(float *inImagekernel, float *outputImagekernel, int inWidth, int inHeight, int outWidth, int outHeight, double theta)

{
	
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float rads = (theta) * 3.1415926 / 180.0;

	float u = (float)col - (float)outWidth / 2;
	float v = (float)row - (float)outHeight / 2;
	float tu = u * cosf(rads) - v * sinf(rads);
	float tv = v * cosf(rads) + u * sinf(rads);

	//tu /= (float)inWidth;
	//tv /= (float)inHeight;
	tu = (float)(tu + inWidth / 2);
	tv = (float)(tv + inHeight / 2);


	if (col < outWidth && row < outHeight && tu >= 0 && tu < inWidth && tv >= 0 && tv < inHeight)
	{
		//outputImagekernel[row*outWidth + col] = inImagekernel[((int)tv * inWidth) + (int)tu];//tex2D(tex_rot_imgA, tu + inWidth/2 , tv + inHeight/2);
		outputImagekernel[row*outWidth + col] = getInterpolatedPixel(tu, tv, inWidth, inHeight, inImagekernel);//tex2D(tex_rot_imgA, tu + inWidth/2 , tv + inHeight/2);
	}
}

//--------------------------------------------------------------------------------
cudaError_t RotateImage_tex_Cuda(float* pInData, int inWidth, int inHeight, float *pOutData, int outWidth, int outHeight, double theta)
{

	cudaArray *cuArray_img;
	float *d_OutData = 0;
	float *d_InData = 0;

	int iFrameSize = inWidth * inHeight;
	int iOutFrameSize = outWidth * outHeight;

	cudaError_t cudaStatus;



	// Choose which GPU to run on, change this on a multi-GPU system.
	//cudaStatus = cudaSetDevice(0);
	//if (cudaStatus != cudaSuccess) {
	//          fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	//          goto Error;
	//}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	//imput image text array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMallocArray(&cuArray_img, &channelDesc, inWidth, inHeight);
	cudaMemcpyToArray(cuArray_img, 0, 0, pInData, iFrameSize * sizeof(float), cudaMemcpyHostToDevice);
	// Set texture parameters
	tex_rot_img.addressMode[0] = cudaAddressModeBorder;// ModeWrap;
	tex_rot_img.addressMode[1] = cudaAddressModeBorder;
	tex_rot_img.filterMode = cudaFilterModeLinear;
	tex_rot_img.normalized = true;    // access with normalized texture coordinates
	cudaBindTextureToArray(tex_rot_img, cuArray_img, channelDesc);

	cudaStatus = cudaMalloc((void**)&d_InData, iFrameSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;

	}
	cudaStatus = cudaMalloc((void**)&d_OutData, iOutFrameSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaMemcpy(d_InData, pInData, iFrameSize * sizeof(float), cudaMemcpyHostToDevice);


//	printf("%d %d %d %d\n", inWidth, inHeight, outWidth, outHeight);

	int TILE_SIZE_X = 16;
	int TILE_SIZE_Y = 16;
	dim3 dimBlock(TILE_SIZE_X, TILE_SIZE_Y);

	dim3 dimGrid((int)ceil((float)outWidth / (float)TILE_SIZE_X), (int)ceil((float)outHeight / (float)TILE_SIZE_Y));

	// Launch a kernel on the GPU with one thread for each element.
	//RotateImage_tex_kernel << <dimGrid, dimBlock >> > (d_OutData, inWidth, inHeight, outWidth, outHeight, theta);
	RotateImage_kernel << <dimGrid, dimBlock >> > (d_InData, d_OutData, inWidth, inHeight, outWidth, outHeight, theta);

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
	cudaStatus = cudaMemcpy(pOutData, d_OutData, iOutFrameSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFreeArray(cuArray_img);
	cudaFree(d_InData);
	cudaFree(d_OutData);


	return cudaStatus;
}


