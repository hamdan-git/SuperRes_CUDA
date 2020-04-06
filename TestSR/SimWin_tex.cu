
#include "Kernel.h"

#include <stdio.h>
#include <math.h>
#include <float.h>


// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex_img;
texture<unsigned char, 2, cudaReadModeElementType> tex_map;

//-------------------------------------------------------------------
__global__ void SimWin_Kernel_tex(float *outputImagekernel, int imageWidth, int imageHeight)

{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	int iOffset = row * imageWidth + col;

	if (tex2D(tex_map, col, row) == 0)
	{
		outputImagekernel[iOffset] = tex2D(tex_img, col, row);// inputImageKernel[iOffset];
		return;
	}
	//return;
	const int iLargeWin = 6;
	const int iWin = 2;


	const int iWinSize = (2 * iWin + 1) * (2 * iWin + 1);

	float pCurValues[iWinSize];
	for (int i = 0; i < iWinSize; i++) 
		pCurValues[i] = -1.0f;

	float fValidSum = 0.0;
	int iIndex = 0;
	int iCurValidNeighbours = 0;
	for (int j = -iWin; j <= iWin; j++)
	{
		for (int i = -iWin; i <= iWin; i++)
		{
			int iNewX = i + col;
			int iNewY = j + row;
			if (iNewX >= 0 && iNewX < imageWidth && iNewY >= 0 && iNewY < imageHeight)
			{
				unsigned char iLabel = tex2D(tex_map, iNewX, iNewY);// ata[iOffset + (j*imageWidth) + i];
				if (iLabel == 0)
				{
					//pCurValues[iIndex] = inputImageKernel[iOffset + (j*imageWidth) + i];
					pCurValues[iIndex] = tex2D(tex_img, iNewX, iNewY);
					iCurValidNeighbours++;
					fValidSum += pCurValues[iIndex];
				}
			}
			iIndex++;
		}
	}



	float fMinDiff = FLT_MAX;

	float fCurentWinMean = 1.0f;
	if (iCurValidNeighbours > 0)
		fCurentWinMean = fValidSum / (double)iCurValidNeighbours;
	int iHalfCurValidNeighbours = (iCurValidNeighbours >> 1) + 1; //added on 10Aug11

	int iStartX = col - iLargeWin + iWin;
	if (iStartX < iWin) iStartX = iWin;
	int iStartY = row - iLargeWin + iWin;
	if (iStartY < iWin) iStartY = iWin;
	int iEndX = iLargeWin + col - iWin;
	if (iEndX >= imageWidth - iWin) iEndX = imageWidth - iWin - 1;
	int iEndY = iLargeWin + row - iWin;
	if (iEndY >= imageHeight - iWin) iEndY = imageHeight - iWin - 1;

	int iTargetX = -1, iTargetY = -1;

	//float *pLocOutData = &outputImagekernel[iStartY*imageWidth + iStartX];
	//unsigned char *pLocMapData = &pMapData[iStartY*imageWidth + iStartX];
	int iJumpX = imageWidth - (iEndX - iStartX + 1);
	for (int jj = iStartY; jj <= iEndY; jj++)
	{
		for (int ii = iStartX; ii <= iEndX; ii++)
		{
			if ((ii != col || jj != row) && (tex2D(tex_map, ii, jj) == 0 )) //ha added on 25Feb2015, cluster size larger than the large win size were not update w/out the last term
			{
				//now search the local win 
				int iIndex = 0;
				int iLocalValidNeighbours = 0;
				double fDiffSum = 0.0;
				//double fSum = 0;
				for (int j = -iWin; j <= iWin; j++)
				{
					for (int i = -iWin; i <= iWin; i++)
					{
						int iNewX = ii + i;
						int iNewY = jj + j;
						if (iNewX >= 0 && iNewX < imageWidth && iNewY >= 0 && iNewY < imageHeight && pCurValues[iIndex] >= 0.0)
						{
							int iLabel = tex2D(tex_map, iNewX, iNewY); ;// *(pLocMapData + (j*imageWidth + i));
							if (iLabel == 0 /*|| iLabel == iLocalLabel*/)
							{
								iLocalValidNeighbours++;
								float pNeiVal = tex2D(tex_img, iNewX,  iNewY);
								double fDiff = fabs((double)(pNeiVal - pCurValues[iIndex]));
								fDiffSum += fDiff;

							}
						}
						iIndex++;
					}
				}
				if (iLocalValidNeighbours >= iHalfCurValidNeighbours && iLocalValidNeighbours > 0) //added on 10Aug11
				{
					fDiffSum /= (double)iLocalValidNeighbours;
					if (fDiffSum < fMinDiff)
					{
						fMinDiff = fDiffSum;
						iTargetX = ii;
						iTargetY = jj;

					}
				}


			}
			//pLocMapData++;
			//pLocOutData++;

		}
		//pLocMapData += iJumpX;
		//pLocOutData += iJumpX;
	}
	if (iTargetX >= 0 && iTargetY >= 0)
	{
		//outputImagekernel[iOffset] = inputImageKernel[iTargetY*imageWidth + iTargetX];// (pOrigData[iTargetY*iWidth + iTargetX]);
		outputImagekernel[iOffset] = tex2D(tex_img, iTargetX, iTargetY);// +iTargetX];// (pOrigData[iTargetY*iWidth + iTargetX]);
	}

}



//--------------------------------------------------------------------------------
cudaError_t SimWin_tex_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, int iNumFrames, unsigned char *pMapData)
{

	//float *d_InData = 0;
	cudaArray *cuArray_img;
	cudaArray *cuArray_map;
	float *d_OutData = 0;

	int iFrameSize = iWidth * iHeight;

	cudaError_t cudaStatus;



	// Choose which GPU to run on, change this on a multi-GPU system.
	//cudaStatus = cudaSetDevice(0);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	//	goto Error;
	//}

// Allocate array and copy image data

	//imput image text array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMallocArray(&cuArray_img, &channelDesc, iWidth, iHeight);
	//cudaMemcpyToArray(cuArray_img, 0, 0, pInData, iFrameSize*sizeof(float), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(tex_img, cuArray_img, channelDesc);

	//defectmap text array
	cudaChannelFormatDesc channelDesc_map = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaMallocArray(&cuArray_map, &channelDesc_map, iWidth, iHeight);
	cudaMemcpyToArray(cuArray_map, 0, 0, pMapData, iFrameSize , cudaMemcpyHostToDevice);
	cudaBindTextureToArray(tex_map, cuArray_map, channelDesc_map);

	cudaStatus = cudaMalloc((void**)&d_OutData, iFrameSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}



	int TILE_SIZE_X = 1;
	int TILE_SIZE_Y = 128;
	dim3 dimBlock(TILE_SIZE_X, TILE_SIZE_Y);

	dim3 dimGrid((int)ceil((float)iWidth / (float)TILE_SIZE_X), (int)ceil((float)iHeight / (float)TILE_SIZE_Y));

	for (int z = 0; z < iNumFrames; z++)
	{
		cudaMemcpyToArray(cuArray_img, 0, 0, &pInData[z*iFrameSize], iFrameSize * sizeof(float), cudaMemcpyHostToDevice);

		// Launch a kernel on the GPU with one thread for each element.
		SimWin_Kernel_tex << <dimGrid, dimBlock >> > (d_OutData, iWidth, iHeight);

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
		cudaStatus = cudaMemcpy(&pOutData[z*iFrameSize], d_OutData, iFrameSize * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}

Error:
	cudaFreeArray(cuArray_img);
	cudaFreeArray(cuArray_map);
	cudaFree(d_OutData);

	return cudaStatus;
}


