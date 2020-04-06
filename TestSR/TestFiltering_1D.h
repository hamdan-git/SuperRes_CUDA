#pragma once
#include "Common.h"
#include<time.h>
#include <stdio.h>

//-------------------------------------------------------
template<typename T>
float *CreateGaussianKernel_1D_Mat(int iSize, float fSigma)
{
	if (iSize <= 0) return NULL;
	if (fSigma <= 0) return NULL;
	float *pKernel = new float[iSize];
	float fHalf = ((float)iSize + 1.0f) / 2.0f;
	float fVal = 0.0f;
	double fSum = 0;
	for (int i = 0; i < iSize; i++)
	{
		fVal = (1 + i) - fHalf;
		fVal = exp(-(fVal*fVal) / (2 * fSigma));
		pKernel[i] = fVal;
		fSum += (double)fVal;
	}
	if (fSum == 0.0)
	{
		delete[] pKernel; pKernel = NULL;
		return NULL;
	}
	for (int i = 0; i < iSize; i++)
	{
		pKernel[i] /= (float)fSum;
	}
	return pKernel;
}

//---------------------------------------------------------------
template<typename T>
void ApplyFiltering_1D_Mat(T *pInData, T *pOutData, int iDataSize, float *pKernel, int iKernelSize)
{
	float *pTempInDataRef = NULL;
	double fSum = 0.0;
	int iWin = iKernelSize / 2;

	for (int i = iWin; i < iDataSize - iWin; i++)
	{
		fSum = 0.0;
		for (int p = -iWin; p <= iWin; p++)
			fSum += pInData[p + i] * pKernel[p + iWin];
		pOutData[i] = (T)fSum;
	}

	//do the first part
	for (int i = 0; i < iWin; i++)
	{
		fSum = 0.0;
		for (int p = 0; p <= iWin; p++)
			fSum += pInData[p + i] * pKernel[iWin + p];
		//take the reverse of data
		for (int p = 1; p <= iWin; p++)
			fSum += pInData[abs(p - i)] * pKernel[iWin - p];
		pOutData[i] = (T)fSum;
	}
	//do the last part
	//for(int i = iDataSize-1; i >= iDataSize-iWin; i--)
	for (int i = 0; i < iWin; i++)
	{
		int iEnd = iDataSize - 1;
		fSum = 0.0;
		for (int p = 0; p <= iWin; p++)
			fSum += pInData[iEnd - (i + p)] * pKernel[iWin + p];

		//take the reverse of data
		for (int p = 1; p <= iWin; p++)
			fSum += pInData[iEnd - abs(p - i)] * pKernel[iWin - p];
		pOutData[iEnd - i] = (T)fSum;
	}



}
//---------------------------------------------------------------
template<typename T>
void ApplyVerticalFiltering_1D_Mat(const T *pInData, T *pOutData, int iWidth, int iHeight, float *pKernel, int iKernelSize)
{
	T *pTempLine = new T[iHeight];
	T *pTempLine2 = new T[iHeight];

	for (int iX = 0; iX < iWidth; iX++)
	{
		long iOffsetY = iX;
		for (int iY = 0; iY < iHeight; iY++)
		{
			pTempLine[iY] = pInData[iOffsetY];
			iOffsetY += iWidth;
		}
		ApplyFiltering_1D_Mat(pTempLine, pTempLine2, iHeight, pKernel, iKernelSize);
		iOffsetY = iX;
		for (int iY = 0; iY < iHeight; iY++)
		{
			pOutData[iOffsetY] = pTempLine2[iY];
			iOffsetY += iWidth;
		}
	}
	delete[] pTempLine;
	delete[] pTempLine2;

}

//--------------------------------------------------------------
void TestFiltering_1D_cpu(float *pData, int iW, int iH, float *pKernel, int size)
{
	int iFrameSize = iW * iH;
	float *pOutData = new float[iFrameSize];

	clock_t start = clock();
	for (int i = 0; i < iIterations; i++)
	{
		ApplyVerticalFiltering_1D_Mat(pData, pOutData, iW, iH, pKernel, size);
	}

	printf("Filter_1D cpu took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("c:\\Temp\\Filter_1D_cpu.raw", pOutData, iW, iH);

	delete[] pOutData;
}


//--------------------------------------------------------------
void TestFiltering_1D_gpu(float *pData, int iW, int iH, float *pKernel, int size)
{
	int iFrameSize = iW * iH;
	float *pOutData = new float[iFrameSize];

	addWithCuda();

	clock_t start = clock();
	for (int i = 0; i < iIterations; i++)
	{
		Filtering_1D_Cuda(pData, pOutData, iW, iH, pKernel, size, 0);
	}

	printf("Filter_1D gpu took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("c:\\Temp\\Filter_1D_gpu.raw", pOutData, iW, iH);

	delete[] pOutData;
}

//--------------------------------------------------------------
void TestFiltering_1D_shared_gpu(float *pData, int iW, int iH, float *pKernel, int size)
{
	int iFrameSize = iW * iH;
	float *pOutData = new float[iFrameSize];

	addWithCuda();

	clock_t start = clock();
	for (int i = 0; i < iIterations; i++)
	{
		Filtering_1D_shared_Cuda(pData, pOutData, iW, iH, pKernel, size, 1);
	}

	printf("Filter_1D gpu_shared took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("c:\\Temp\\Filter_1D_shared_gpu.raw", pOutData, iW, iH);

	delete[] pOutData;
}

//--------------------------------------------------------------
void TestFiltering_1D_tex_gpu(float *pData, int iW, int iH, float *pKernel, int size)
{
	int iFrameSize = iW * iH;
	float *pOutData = new float[iFrameSize];

	addWithCuda();

	clock_t start = clock();
	for (int i = 0; i < iIterations; i++)
	{
		Filtering_1D_tex_Cuda(pData, pOutData, iW, iH, pKernel, size, 0);
	}

	printf("Filter_1D gpu_shared_tex took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("c:\\Temp\\Filter_1D_shared_tex_gpu.raw", pOutData, iW, iH);

	delete[] pOutData;
}
//cudaError_t Filtering_1D_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, float *pKernel, int iKernelSize, int dir)

//--------------------------
void TestFiltering_1D()
{
	int iWidth = 2048;// 1024;
	int iHeight = 2048;// 256;
	float *pInImage = ReadRawData<float>("C:\\Temp\\TestMedian.raw", iWidth, iHeight);
	if (pInImage == NULL) { printf("null input"); exit(0); }

	int iKernelSize = 9;
	float fSigma = 7;
	float *pKernel = CreateGaussianKernel_1D_Mat<float>(iKernelSize, fSigma);

	TestFiltering_1D_cpu(pInImage, iWidth, iHeight, pKernel, iKernelSize);
	TestFiltering_1D_gpu(pInImage, iWidth, iHeight, pKernel, iKernelSize);
	TestFiltering_1D_shared_gpu(pInImage, iWidth, iHeight, pKernel, iKernelSize);
	TestFiltering_1D_tex_gpu(pInImage, iWidth, iHeight, pKernel, iKernelSize);
	//	TestInterpolateBetween_gpu(pInImage, iWidth, iHeight);

	delete[] pKernel;
	delete[] pInImage;
}