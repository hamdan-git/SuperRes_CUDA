#include "Kernel.h"

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "IPTools.h"
#include <math.h>
#include <algorithm>
#include "Common.h"

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif
#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex_rot_imgA;


//-----------------------------------------------------------------------------
__device__ double getInterpolatedPixelA(double x, double y, int iWidth, int iHeight, float* pixels)
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
__global__ void RotateImageA_tex_kernel(float *outputImagekernel, int inWidth, int inHeight, int outWidth, int outHeight, double theta)

{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float rads = (theta) * 3.1415926 / 180.0;

	float u = (float)col - (float)outWidth / 2;
	float v = (float)row - (float)outHeight / 2;
	float tu = u * cosf(rads) - v * sinf(rads);
	float tv = v * cosf(rads) + u * sinf(rads);

	tu /= (float)inWidth;
	tv /= (float)inHeight;



	if (col < outWidth && row < outHeight)
		outputImagekernel[row*outWidth + col] = tex2D(tex_rot_imgA, tu + 0.5f, tv + 0.5f);
}

__global__ void RotateImageA_kernel(float *inImagekernel, float *outputImagekernel, int inWidth, int inHeight, int outWidth, int outHeight, double theta)

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
		outputImagekernel[row*outWidth + col] = inImagekernel[((int)tv * inWidth) + (int)tu];//tex2D(tex_rot_imgA, tu + inWidth/2 , tv + inHeight/2);
		//outputImagekernel[row*outWidth + col] = getInterpolatedPixelA(tu, tv, inWidth, inHeight, inImagekernel);
	}
}

__global__ void SmoothBorder_kernel(float *inImagekernel, float *outputImagekernel, unsigned char *pMaskData, int iWidth, int iHeight, int iWin)
{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int iOffset = row * iWidth + col;
	outputImagekernel[iOffset] = inImagekernel[iOffset] * inImagekernel[iOffset];
	//if (pMaskData[iOffset] > 0)
	//{
	//	int count = 0;
	//	double fSum = 0.0;
	//	for (int j = -iWin; j <= iWin; j++)
	//	{
	//		for (int i = -iWin; i <= iWin; i++)
	//		{
	//			int iNewX = col + i;
	//			int iNewY = row + j;
	//			if (iNewX >= 0 && iNewX < iWidth && iNewY >= 0 && iNewY < iHeight)
	//			{
	//				fSum += (double)inImagekernel[iNewY*iWidth + iNewX];
	//				count++;
	//			}
	//		}
	//	}
	//	if (count > 0)
	//	{
	//		outputImagekernel[iOffset] = (float)(fSum / (double)count);
	//	}
	//}
}
//-------------------------------------------------------------------
__global__ void TDS_AddA_kernel(float *inFrameData,  float *outputImageData, int inWidth, int inHeight)

{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ( col < inWidth && row < inHeight)
		outputImageData[row*inWidth + col] += inFrameData[row*inWidth + col];
}

//-------------------------------------------------------------------
__global__ void SetValues_kernel(float *inFrameData, float iVal, int inWidth, int inHeight)

{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < inWidth && row < inHeight)
		inFrameData[row*inWidth + col] = iVal;
}

//-------------------------------------------------------------------
__global__ void Memcpy_us_to_float__kernel(unsigned short *pInData, float *pOutData, int inWidth, int inHeight)

{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < inWidth && row < inHeight)
		pOutData[row*inWidth + col] = pInData[row*inWidth + col];
}

//----------------------------------------------------------------------------
unsigned char *GetMaskDataAfterRotation(int iW, int iH, double theta, int &iNewW, int &iNewH)
{
	unsigned char *pMaskData = NULL;
	//int iNewW=0, iNewH = 0;
	FindDimensionAfterRotation(iW, iH, theta, iNewW, iNewH);
	if (iNewW <= 0 || iNewH <= 0) return NULL;
	pMaskData = new unsigned char[iNewW*iNewH];
	unsigned char *pTempData = new unsigned char[iW*iH];
	unsigned char *pRotatedTempData = new unsigned char[iNewW*iNewH];
	memset(pRotatedTempData, 0, iNewW*iNewH);

	for (long i = 0; i < iW*iH; i++)
		pTempData[i] = 1;

	unsigned char *pMaks_rot_eroded = new unsigned char[iNewW*iNewH];
	memset(pMaks_rot_eroded, 0, iNewW*iNewH);
	IPTools<unsigned char>::RotateImage_cpu(pTempData, iW, iH, pRotatedTempData, iNewW, iNewH, theta, 0);
	IPTools<unsigned char>::DoErosion(pRotatedTempData, pMaks_rot_eroded, iNewW, iNewH, 5);
	float *pDisData = new float[iNewW*iNewH];
	IPTools<unsigned char>::GetDistanceMap(pMaks_rot_eroded, iNewW, iNewH, 0, pDisData);

	//WriteRawData<unsigned char>("c:\\Temp\\Dist.raw", pMaks_rot_eroded, iNewW, iNewH);

	memset(pMaskData, 0, 1 * iNewW*iNewH);
	for (long i = 0; i < iNewW*iNewH; i++)
	{
		float fVal = pDisData[i];
		if (fVal > 0.0f && fVal < 6.0f)
			pMaskData[i] = 1;
	}

	delete[] pTempData;
	delete[] pRotatedTempData;
	delete[] pMaks_rot_eroded;
	delete[] pDisData;
	return pMaskData;
}

//--------------------------------------------------------------------------------
cudaError_t RotateAddImage_tex_Cuda(unsigned short* pInData, int inWidth, int inHeight, int iNumFrames, unsigned short *pOutData, int outWidth, int outHeight, double theta, double fScale)
{

	cudaArray *cuArray_img;
	float *d_OutData = 0;
	unsigned short *d_InData_us = 0;
	float *d_InData = 0;
	float *d_RotatedFrameData = 0;
	unsigned char *d_pMaskData = 0;
	float *pTempInData = 0;
	float *pTempOutData = 0;

	int iFrameSize = inWidth * inHeight;
	int iOutFrameSize = outWidth * outHeight;

	pTempInData = new float[iFrameSize];
	pTempOutData = new float[iOutFrameSize];

	cudaError_t cudaStatus = cudaErrorInvalidValue;

	int iRotWidth, int iRotHeight;
	FindDimensionAfterRotation(inWidth, inHeight, theta, iRotWidth, iRotHeight);

	int iRotatedFrameSize = iRotWidth * iRotHeight;

	//Get mask data to cover arround the edges after rotation
	int iNewMaskW = 0, iNewMaskH = 0;
	unsigned char *pMaskData = GetMaskDataAfterRotation(inWidth, inHeight, theta, iNewMaskW , iNewMaskH);
	if (iNewMaskW != iRotWidth || iNewMaskH != iRotHeight)
	{
		printf("dimension mismatch when creating mask\n");
		goto Error;
	}
	//WriteRawData<unsigned char>("c:\\Temp\\MaskData.raw", pMaskData, iNewMaskW, iNewMaskH);

	printf("Rotaed dim %d %d %d %d\n", iRotWidth, iRotHeight, outWidth,  outHeight);
	// Choose which GPU to run on, change this on a multi-GPU system.
	//cudaStatus = cudaSetDevice(0);
	//if (cudaStatus != cudaSuccess) {
	//          fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	//          goto Error;
	//}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	//imput image text array
//	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
//	cudaMallocArray(&cuArray_img, &channelDesc, inWidth, inHeight);
////	cudaMemcpyToArray(cuArray_img, 0, 0, pInData, iFrameSize * sizeof(unsigned short), cudaMemcpyHostToDevice);
//	// Set texture parameters
//	tex_rot_imgA.addressMode[0] = cudaAddressModeBorder;// ModeWrap;
//	tex_rot_imgA.addressMode[1] = cudaAddressModeBorder;
//	tex_rot_imgA.filterMode = cudaFilterModeLinear;
//	tex_rot_imgA.normalized = true;    // access with normalized texture coordinates
//	cudaBindTextureToArray(tex_rot_imgA, cuArray_img, channelDesc);

	cudaStatus = cudaMalloc((void**)&d_OutData, iOutFrameSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_InData_us, iFrameSize * sizeof(unsigned short));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_InData, iFrameSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_RotatedFrameData, iRotatedFrameSize * sizeof(float)*2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
		
	cudaStatus = cudaMalloc((void**)&d_pMaskData, iRotatedFrameSize * sizeof(unsigned char) );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaMemcpy(d_pMaskData, pMaskData, iRotatedFrameSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

	int TILE_SIZE_X = 16;
	int TILE_SIZE_Y = 16;
	dim3 dimBlock(TILE_SIZE_X, TILE_SIZE_Y);

	dim3 dimGrid((int)ceil((float)iRotWidth / (float)TILE_SIZE_X), (int)ceil((float)iRotHeight / (float)TILE_SIZE_Y));
	
	dim3 dimGrid_in((int)ceil((float)inWidth / (float)TILE_SIZE_X), (int)ceil((float)inHeight / (float)TILE_SIZE_Y));

	float fShiftRow = 0.0f;
	bool bReversed = false;
	int iCurIndex = 0;
	int iPrevIndex = -1;
	for (int iZ = 0; iZ < iNumFrames  ; iZ++)
	{
		iCurIndex = (int)fShiftRow*outWidth;
		if (iPrevIndex != iCurIndex)
		{
			int iZIndex = bReversed ? iZ : iNumFrames - 1 - iZ;

			//////////////////
			//unsigned short *pInDataRef = &pInData[iZIndex * iFrameSize];
			//for (int k = 0; k < iFrameSize; k++)
			//	pTempInData[k] = (float)pInDataRef[k];
			//cudaMemcpy(d_InData, pTempInData, iFrameSize * sizeof(float), cudaMemcpyHostToDevice);
			/////////////////////

			cudaMemcpy(d_InData_us, &pInData[iZIndex * iFrameSize], iFrameSize * sizeof(unsigned short), cudaMemcpyHostToDevice);
			Memcpy_us_to_float__kernel << <dimGrid_in, dimBlock>> > (d_InData_us, d_InData, inWidth, inHeight);

			SetValues_kernel << <dimGrid, dimBlock >> > (d_RotatedFrameData, 0, iRotWidth, iRotHeight);
			RotateImageA_kernel << <dimGrid, dimBlock >> > (d_InData, &d_RotatedFrameData[0], inWidth, inHeight, iRotWidth, iRotHeight, theta);
			SmoothBorder_kernel << <dimGrid, dimBlock >> > (&d_RotatedFrameData[0], &d_RotatedFrameData[iRotatedFrameSize], d_pMaskData, iRotWidth, iRotHeight, 3);
			TDS_AddA_kernel << <dimGrid, dimBlock >> > (&d_RotatedFrameData[0], &d_OutData[iCurIndex], iRotWidth, iRotHeight);
			
			cudaDeviceSynchronize();
		}
		fShiftRow += fScale;
		iPrevIndex = iCurIndex;
		if (fShiftRow >= outHeight) break;
	}

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
	//cudaStatus = cudaMemcpy(pOutData, d_OutData, iOutFrameSize * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(pTempOutData, d_OutData, iOutFrameSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	for (int k = 0; k < iOutFrameSize; k++)
		pOutData[k] = (unsigned short)pTempOutData[k];
	
Error:
	//cudaFreeArray(cuArray_img);
	if ( d_InData!=NULL )cudaFree(d_InData);
	if (d_OutData!=NULL) cudaFree(d_OutData);
	if (d_RotatedFrameData!=NULL) cudaFree(d_RotatedFrameData);
	if (d_pMaskData!=NULL)cudaFree(d_pMaskData);
	if (d_InData_us != NULL)cudaFree(d_InData_us);
	delete[] pMaskData;
	delete[] pTempInData;
	delete[] pTempOutData;
	return cudaStatus;
}


