#include "Kernel.h"

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <vector>

// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex_rot_img;



//-------------------------------------------------------------------
__global__ void Memset_kernel(float *pInData, float fVal, int iWidth)

{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	pInData[row*iWidth + col] = fVal;
}

//-------------------------------------------------------------------
__global__ void InterpolateBetweenPoints_kernel (float *pInData, float *pOutData, float *pRef1, float *pRef2, float *pGain1, float *pGain2, int iWidth, int iHeight, int iRefPointIndex, int iNumPoints, float fOneOver_s)
{

	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int iFrameSize = iWidth * iHeight;
	unsigned int iOffset = row * iWidth + col;

	float fInVal = pInData[iOffset];
	float fInScaledVal = fInVal * fOneOver_s;

	double fDiff = (double)(pRef2[iOffset] - pRef1[iOffset]);// + 0.001;
	if (fDiff == 0.0) fDiff = 0.001; //small value

	double fBeta = (double)(fInScaledVal - pRef1[iOffset]) / fDiff;
	double fAlpha = 1.0 - fBeta;
	double fOverShoot = 0.2;

	if (iRefPointIndex == 0)
	{
		if (fAlpha > (1.2)) fAlpha = 1.2;
		if (fBeta < -fOverShoot) fBeta = -fOverShoot;
	}
	else
	{
		if (fAlpha > 1.0) fAlpha = 0.0;
		if (fBeta < 0) fBeta = 0.0;
	}

	if (iRefPointIndex == iNumPoints - 1)
	{
		if (fAlpha < -fOverShoot) fAlpha = -fOverShoot;
		if (fBeta > (1.2)) fBeta = 1.2;
	}
	else
	{
		if (fAlpha < 0) fAlpha = 0.0;
		if (fBeta >= 1.0) fBeta = 0.0;
	}

	double fInterpolated = fAlpha * pGain1[iOffset] + fBeta * pGain2[iOffset];
	pOutData[iOffset] += fInterpolated * fInVal;


}

//-------------------------------------------------------------------
__global__ void InterpolateBetweenPoints_kernel2(float *pInData, float *pOutData, float *pGain, long iRefIndex1, long iRefIntex2, long iGainIndex1, long iGainIndex2, int iWidth, int iHeight, int iRefPointIndex, int iNumPoints, float fOneOver_s)
{

	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int iFrameSize = iWidth * iHeight;
	unsigned int iOffset = row * iWidth + col;

	float *pRef1 = &pGain[iRefIndex1+ iOffset];
	float *pRef2 = &pGain[iRefIntex2 + iOffset];
	float *pGain1 = &pGain[iGainIndex1 + iOffset];
		float *pGain2 = &pGain[iGainIndex2 + iOffset];

	float fInVal = pInData[iOffset];
	float fInScaledVal = fInVal * fOneOver_s;

	double fDiff = (double)(pRef2[iOffset] - pRef1[iOffset]);// + 0.001;
	if (fDiff == 0.0) fDiff = 0.001; //small value

	double fBeta = (double)(fInScaledVal - pRef1[iOffset]) / fDiff;
	double fAlpha = 1.0 - fBeta;
	double fOverShoot = 0.2;

	if (iRefPointIndex == 0)
	{
		if (fAlpha > (1.2)) fAlpha = 1.2;
		if (fBeta < -fOverShoot) fBeta = -fOverShoot;
	}
	else
	{
		if (fAlpha > 1.0) fAlpha = 0.0;
		if (fBeta < 0) fBeta = 0.0;
	}

	if (iRefPointIndex == iNumPoints - 1)
	{
		if (fAlpha < -fOverShoot) fAlpha = -fOverShoot;
		if (fBeta > (1.2)) fBeta = 1.2;
	}
	else
	{
		if (fAlpha < 0) fAlpha = 0.0;
		if (fBeta >= 1.0) fBeta = 0.0;
	}

	double fInterpolated = fAlpha * pGain1[iOffset] + fBeta * pGain2[iOffset];
	pOutData[iOffset] += fInterpolated * fInVal;
}


//--------------------------------------------------------------------------------
cudaError_t InterpolateBetweenFilterPoints_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, int iNumFrames, std::vector<float*> *pGain, int iNumPoints, int iNumBins, float fps)
{

	//cudaArray *cuArray_img;
	float *d_InData = 0;
	float *pTempInput = 0;
	float *d_OutData = 0;
	float *d_pGain = 0;


	int iFrameSize = iWidth * iHeight;
	long iGainImagesSize = iFrameSize * 2  *iNumBins;
	float fOneOver_s = 100;

	cudaError_t cudaStatus;

	pTempInput = new float[iFrameSize];

	// Choose which GPU to run on, change this on a multi-GPU system.
	//cudaStatus = cudaSetDevice(0);
	//if (cudaStatus != cudaSuccess) {
	//          fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	//          goto Error;
	//}

	cudaStatus = cudaMalloc((void**)&d_InData, iFrameSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_pGain, iGainImagesSize * iNumPoints * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_OutData, iFrameSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//put the gain image
	for (int k = 0; k < iNumPoints; k++)
	{
		cudaStatus = cudaMemcpy(&d_pGain[k*iGainImagesSize], pGain->at(k), iGainImagesSize * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Gain cudaMemcpy failed!");
			goto Error;
		}
	}

	int TILE_SIZE_X = 16;
	int TILE_SIZE_Y = 16;
	dim3 dimBlock(TILE_SIZE_X, TILE_SIZE_Y);

	dim3 dimGrid((int)ceil((float)iWidth / (float)TILE_SIZE_X), (int)ceil((float)iHeight / (float)TILE_SIZE_Y));

	
	for (int iZ = 0; iZ < iNumFrames; iZ++)
	{
		double t = (double)(iZ)* fps;
		//iBinIndex= (int)(std::ceil(t/binSize));
		int iBinIndex = (int)t;// (int)((float)t / (float)binSize);
		iBinIndex = iBinIndex < 0 ? 0 : iBinIndex;//  max(iBinIndex, 0);
		iBinIndex = iBinIndex > iNumBins - 1 ? iNumBins - 1 : iBinIndex;// min(iBinIndex, iNumBins - 1);

		// Copy input vectors from host memory to GPU buffers.
		for (long m = 0; m < iFrameSize; m++)
			pTempInput[m] = (float)pInData[iZ*iFrameSize + m];
		//cudaStatus = cudaMemcpy(d_InData, &pInData[iZ*iFrameSize], iFrameSize * sizeof(float), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(d_InData, pTempInput, iFrameSize * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Input cudaMemcpy failed!");
			goto Error;
		}

		//reset output image
		Memset_kernel << < dimGrid, dimBlock >> > (d_OutData, 0.0f, iWidth);
		cudaDeviceSynchronize();

		for (int k = 0; k < iNumPoints - 1; k++)
		{
			long iRefIndex1 =  k * iGainImagesSize + iBinIndex * iFrameSize;
			long iRefIndex2 =  (k + 1) * iGainImagesSize + iBinIndex * iFrameSize;
			long iGainIndex1 =  iRefIndex1 + (iFrameSize*iNumBins);
			long iGainIndex2 =  iRefIndex2 + (iFrameSize*iNumBins);

			//call trhe kernel 
			InterpolateBetweenPoints_kernel << <dimGrid, dimBlock >> > (d_InData, d_OutData, &d_pGain[iRefIndex1], &d_pGain[iRefIndex2], &d_pGain[iGainIndex1], &d_pGain[iGainIndex2], iWidth, iHeight, k, iNumPoints-1, fOneOver_s);
			//InterpolateBetweenPoints_kernel2 << <dimGrid, dimBlock >> > (d_InData, d_OutData, d_pGain, iRefIndex1, iRefIndex2, iGainIndex1, iGainIndex2, iWidth, iHeight, k, iNumPoints, fOneOver_s);

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
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching InterpolateBetweenPoints_kernel!\n", cudaStatus);
				fprintf(stderr, "Occured at z=%d and k=%d, binIndex=%d, refIndex1=%d and refIndex2=%d!\n", iZ, k, iBinIndex, iRefIndex1, iRefIndex2);
				goto Error;
			}

		}
		
		cudaStatus = cudaMemcpy(&pOutData[iZ*iFrameSize], d_OutData, iFrameSize * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "output cudaMemcpy failed!");
			goto Error;
		}

	}
	



Error:
	//	cudaFreeArray(cuArray_img);
	cudaFree(d_OutData);
	cudaFree(d_InData);
	cudaFree(d_pGain);




	return cudaStatus;
}
