#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "Kernel.h"
#include <ipp.h>
#include <ipps.h>
#include <Windows.h>
#include <vector>
#include <algorithm>
#include "Common.h"
#include "TestInterpolateBetweenCols.h"
#include "TestFiltering_1D.h"
#include "TestSimilarityWindow.h"
#include "TestRotateImage.h"
#include "TestRotateAddImage.h"
#include "TestInterpolateBetweenFilterPoints.h"

//int iIterations = 10;

//--------------------------------------------------------------

//-------------------------------------------------------------------
template <typename T>
bool GetHorizontalMedianFiltering_MCP_UsingIPP_2(const T *pInImage, T *pOutImage, int iWidth, int iHeight, int iStartY, int iEndY, int iWin)
{
	try
	{
		memcpy(pOutImage, pInImage, sizeof(T)* iHeight*iWidth);

		if (iEndY >= iHeight || iStartY < 0) return false;

		//Ipp32f pSrc[1000];
		//int iSkip = 1; //ha added
		//std::vector<float> vValues(2*iWin+iHeight); //plus padded
		Ipp32f *vValues = ippsMalloc_32f(2 * iWin + iWidth);
		Ipp32f *vDestValues = ippsMalloc_32f(2 * iWin + iWidth);
		//std::vector<float> vSorted((2 * iWin + 1));
		int iMaskSize = 2 * iWin + 1;
		//ippsFilterMedianGetBufferSize(iMaskSize, ipp32f, &iBufSize);
		Ipp8u buffWork[10000];
		for (int iY = iStartY; iY <= iEndY; iY++)
		{
			//we need to pad the line with mean of start and end
			const T *pInDataRef = &pInImage[iY*iWidth];
			double fSum = 0;
			int iCount = 0;
			for (int iX = 0; iX < iWin && iX < iWidth; iX++)
			{
				fSum += pInDataRef[iX];
				iCount++;
			}
			float fMean = 1.0f;
			if (iCount > 0) fMean = (float)(fSum / (double)iCount);
			for (int i = 0; i < iWin; i++) vValues[i] = fMean;

			fSum = 0;
			iCount = 0;
			for (int iX = iWidth - 1; iX > iWidth - iWin - 1 && iX >= 0; iX--)
			{
				fSum += pInDataRef[iX];
				iCount++;
			}
			fMean = 1.0f;
			if (iCount > 0) fMean = (float)(fSum / (double)iCount);
			for (int i = iWidth + iWin; i < 2 * iWin + iWidth; i++)
				vValues[i] = fMean;

			for (int iX = iWin; iX < iWidth + iWin; iX++)
			{
				vValues[iX] = pInDataRef[iX];
			}

			IppStatus ipStat = ippsFilterMedian_32f(vValues, vDestValues, 2 * iWin + iWidth, iMaskSize, nullptr, nullptr, buffWork);
			memcpy(&pOutImage[iY*iWidth], &vDestValues[2 * iWin], sizeof(T)*iWidth);
		}


		ippsFree(vValues);
		ippsFree(vDestValues);

	}
	catch (...)
	{
	}

	return true;
}


//-------------------------------------------------------------------
template <typename T>
bool GetHorizontalMedianFiltering_MCP_cpu(const T *pInImage, T *pOutImage, int iWidth, int iHeight, int iStartY, int iEndY, int iWin)
{
	try
	{
		memcpy(pOutImage, pInImage, sizeof(T)* iHeight*iWidth);

		if (iEndY >= iHeight || iStartY < 0) return false;

		//Ipp32f pSrc[1000];
		//int iSkip = 1; //ha added
		//std::vector<float> vValues(2*iWin+iHeight); //plus padded
		Ipp32f *vValues = ippsMalloc_32f(2 * iWin + iWidth);
		Ipp32f *vDestValues = ippsMalloc_32f(2 * iWin + iWidth);
		//std::vector<float> vSorted((2 * iWin + 1));
		int iMaskSize = 2 * iWin + 1;
		//ippsFilterMedianGetBufferSize(iMaskSize, ipp32f, &iBufSize);
		Ipp8u buffWork[10000];
		for (int iY = iStartY; iY <= iEndY; iY++)
		{
			//we need to pad the line with mean of start and end
			const T *pInDataRef = &pInImage[iY*iWidth];
			double fSum = 0;
			int iCount = 0;
			for (int iX = 0; iX < iWin && iX < iWidth; iX++)
			{
				fSum += pInDataRef[iX];
				iCount++;
			}
			float fMean = 1.0f;
			if (iCount > 0) fMean = (float)(fSum / (double)iCount);
			for (int i = 0; i < iWin; i++) vValues[i] = fMean;

			fSum = 0;
			iCount = 0;
			for (int iX = iWidth - 1; iX > iWidth - iWin - 1 && iX >= 0; iX--)
			{
				fSum += pInDataRef[iX];
				iCount++;
			}
			fMean = 1.0f;
			if (iCount > 0) fMean = (float)(fSum / (double)iCount);
			for (int i = iWidth + iWin; i < 2 * iWin + iWidth; i++)
				vValues[i] = fMean;
			////ha
			std::vector<float> vList(2*iWin+1);
			for (int iX = iWin; iX < iWidth + iWin; iX++)
			{
				for (int k = -iWin; k <= iWin; k++)
					vList[k + iWin] = pInDataRef[iX + k];
				std::sort(vList.begin(), vList.end());
				pOutImage[iY*iWidth + (iX - iWin)] = vList[vList.size() / 2];
			}

		}


		ippsFree(vValues);
		ippsFree(vDestValues);

	}
	catch (...)
	{
	}

	return true;
}


//------------------------------------------------------------------
void TestHorizontalMedianFilter_IPP(float *pData, int iW, int iH)
{
	int iFrameSize = iW * iH;
	float *pOutData = new float[iFrameSize];

	clock_t start = clock();
	for (int i = 0; i < iIterations; i++)
	{
		GetHorizontalMedianFiltering_MCP_UsingIPP_2(pData, pOutData, iW, iH, 0, iH - 1, 20);
	}

	printf("IPP took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("E:\\Temp\\IPPRes.raw", pOutData, iW, iH);

	delete[] pOutData;
}

//------------------------------------------------------------------
void TestHorizontalMedianFilter_cpu(float *pData, int iW, int iH, int iWin)
{
	int iFrameSize = iW * iH;
	float *pOutData = new float[iFrameSize];

	clock_t start = clock();
	for (int i = 0; i < iIterations; i++)
	{
		GetHorizontalMedianFiltering_MCP_cpu(pData, pOutData, iW, iH, 0, iH - 1, 20);
	}

	printf("cpu took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("E:\\Temp\\CPURes.raw", pOutData, iW, iH);

	delete[] pOutData;
}


//---------------------------------------------------------------
void TestHorizontalMedianFilterCuda(float *pInData, int iW, int iH, int iWin)
{
	SetCudaDevice(0);

	float *pOutData = new float[iW*iH];

	//warup
	//HorizontalMedianFilterCuda(pInData, pOutData, iW, iH, 20);
	addWithCuda();

	clock_t start = clock();

	for (int i = 0; i < iIterations; i++)
	{
		HorizontalMedianFilterCuda(pInData, pOutData, iW, iH, 20, 0);
	}

	printf("horizontal cuda took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("E:\\Temp\\cuda_horizontal.raw", pOutData, iW, iH);

	delete[] pOutData;
}

//---------------------------------------------------------------
void TestVerticalMedianFilterCuda(float *pInData, int iW, int iH, int iWin)
{
	SetCudaDevice(0);

	float *pOutData = new float[iW*iH];

	//warup
	//HorizontalMedianFilterCuda(pInData, pOutData, iW, iH, 20);
	addWithCuda();

	clock_t start = clock();

	for (int i = 0; i < iIterations; i++)
	{
		HorizontalMedianFilterCuda(pInData, pOutData, iW, iH, 20, 1);
	}

	printf("vertical cuda took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("E:\\Temp\\cuda_vertical.raw", pOutData, iW, iH);

	delete[] pOutData;
}

void quicksort_10(float *number, int first, int last) {
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
		quicksort_10(number, first, j - 1);
		quicksort_10(number, j + 1, last);

	}
}

//------------------------------------------------------------
void NormalSort(float *pData, int size)
{
	for (int i = 0; i < size; i++)
	{
		float fCur = pData[i];
		for (int j = i + 1; j < size; j++)
		{
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

//------------------------------------------------------------
void NormalSort2(float *pData, int size)
{
	for (int i = 0; i < size/2; i++)
	{
		int minval = i;
		for (int j = i + 1; j < size; j++)
		{
			if (pData[j] < pData[minval])
				minval = j;
		}
		{

				//Swap the variables.
				float tmp = pData[i];
				pData[i] = pData[minval];
				pData[minval] = tmp;

			
		}
	}
}

//------------------------------------------------------------
void BubbleSort(float *pData, int size)
{
	int i, j;
	//bool swapped;
	for (i = 0; i < size - 1; i++)
	{
		//swapped = false;
		for (j = 0; j < size - i - 1; j++)
		{
			if (pData[j] > pData[j + 1])
			{
				float tmp = pData[j];
				pData[j] = pData[j + 1];
				pData[j + 1] = tmp;
				//swapped = true;
			}
		}

		// IF no two elements were swapped by inner loop, then break 
		//if (swapped == false)
			//break;
	}
}
//-------------------------------------------------------
void TestSortingAlgorithm()
{
	int size = 10000;
	float *pData = new float[size];
	float *pNormalSort = new float[size];
	float *pQuickSort = new float[size];
	for (int i = 0; i < size; i++)
	{
		pData[i] = rand() % 1000 * 0.25f;
	}
	memcpy(pNormalSort, pData, sizeof(float)*size);
	memcpy(pQuickSort, pData, sizeof(float)*size);
	int ite = 1;
	clock_t start = clock();
	for (int i = 0; i < ite; i++)
	{
		NormalSort(pNormalSort, size);
	}
	float fTime1 = (float)(clock() - start) / (float)CLOCKS_PER_SEC;


	start = clock();
	for (int i = 0; i < ite; i++)
	{
		//quicksort_10(pQuickSort, 0, size - 1);
		//BubbleSort(pQuickSort, size);
		NormalSort2(pQuickSort, size);
	}
	float fTime2 = (float)(clock() - start) / (float)CLOCKS_PER_SEC;

	int iDiffCount = 0;
	for (int i = 0; i < size; i++)
	{
		if (pNormalSort[i] != pQuickSort[i]) iDiffCount++;
	}
	
	printf( "%f %f  %d %f %f\n", fTime1, fTime2, iDiffCount, pNormalSort[1], pQuickSort[1]);
	
}

//---------------------------------------------------------
template<typename T>
void TestGradientMean(T *pInData, int iW, int iH)
{

	int iFrameSize = iW * iH;

	T *pOutData = new T[iFrameSize];
	memcpy(pOutData, pInData, sizeof(T)*iFrameSize);

	float *pNorm = new float[9];

	for (int iY = 1; iY < iH - 1; iY++)
	{
		for (int iX = 1; iX < iW - 1; iX++)
		{
			int index = 0;
			double fMax = 0.0;
			for (int j = -1; j <= 1; j++)
			{
				for (int i = -1; i <= 1; i++)
				{
					float fVal = pInData[((iY + j)*iW) + (iX + i)];
					pNorm[index++] = fVal;
					fMax = fVal > fMax ? fVal : fMax;
				}
			}
			if (fMax > 0.0)
			{
				for (int i = 0; i < 9; i++) pNorm[i] /= fMax;
			}
			float fD_TB = fabs(pNorm[7] - pNorm[1]);//-2*pNorm[4]);
			float fD_LR = fabs(pNorm[3] - pNorm[5]);//-2*pNorm[4]);
			float fD_TlBr = fabs(pNorm[0] - pNorm[8]);//-2*pNorm[4]);
			float fD_TrBl = fabs(pNorm[2] - pNorm[6]);//-2*pNorm[4]);
			float fSum = fD_TB + fD_LR + fD_TlBr + fD_TrBl;
			if (fSum > 0.0f)
			{
				fD_TB /= fSum;
				fD_LR /= fSum;
				fD_TlBr /= fSum;
				fD_TrBl /= fSum;
			}
			fD_TB = 1.0f - fD_TB;
			fD_LR = 1.0f - fD_LR;
			fD_TlBr = 1.0f - fD_TlBr;
			fD_TrBl = 1.0f - fD_TrBl;
			 fSum = fD_TB + fD_LR + fD_TlBr + fD_TrBl;
			if (fSum > 0.0f)
			{
				fD_TB /= fSum;
				fD_LR /= fSum;
				fD_TlBr /= fSum;
				fD_TrBl /= fSum;
			}
			int iOffset = iY * iW + iX;
			float fR = (fD_TB)*(pInData[iOffset - iW] + pInData[iOffset + iW]) + (fD_LR)*(pInData[iOffset - 1] + pInData[iOffset + 1]);
			fR += (fD_TlBr)*(pInData[iOffset - iW - 1] + pInData[iOffset + iW + 1]) + (fD_TrBl)*(pInData[iOffset - iW + 1] + pInData[iOffset + iW - 1]);
			pOutData[iOffset] = fR * 0.5f;
		}
	}
	WriteRawData<T>("c:\\temp\\GradMean.raw", pOutData, iW, iH);
}

//----------------------------------------------------------------------
/** Uses bilinear interpolation to find the pixel value at real coordinates (x,y). */
template<typename T>
double getInterpolatedPixel2_TF(double x, double y, int iWidth, int iHeight, T* pixels)
{
	int xbase = (int)x;
	int ybase = (int)y;

	double xFraction = x - xbase;
	double yFraction = y - ybase;
	int offset = ybase * iWidth + xbase;
	double lowerLeft = 0.0;
	double lowerRight = 0.0;
	double upperRight = 0.0;
	double upperLeft = 0.0;
	bool bOutOfBound = false;
	if (xbase >= 0 && xbase < iWidth && ybase >= 0 && ybase < iHeight)
		lowerLeft = pixels[ybase*iWidth + xbase];
	else
		bOutOfBound = true;
	if ((xbase + 1) >= 0 && (xbase + 1) < iWidth && ybase >= 0 && ybase < iHeight)
		lowerRight = pixels[ybase*iWidth + xbase + 1];
	else
		bOutOfBound = true;
	if ((xbase + 1) >= 0 && (xbase + 1) < iWidth && (ybase + 1) >= 0 && (ybase + 1) < iHeight)
		upperRight = pixels[(ybase + 1)*iWidth + xbase + 1];
	else
		bOutOfBound = true;
	if ((xbase) >= 0 && (xbase) < iWidth && (ybase + 1) >= 0 && (ybase + 1) < iHeight)
		upperLeft = pixels[(ybase + 1)*iWidth + xbase];
	else
		bOutOfBound = true;
	if (bOutOfBound)
	{
		return (lowerLeft + lowerRight + upperRight + upperLeft) / 4.0;
	}
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


//---------------------------------------------------------------------------
void main(int argc,  char *argv[])
{

	//check if we have a device first
	int nDevices = 0;
	cudaGetDeviceCount(&nDevices);
	if (nDevices <= 0)
	{
		printf("No CUDA device was found\n");
		exit(0);
	}
	printf("Number of devices found: %d\n", nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}
	printf("We are going to use device 0\n");
	

//	TestInterpolateBetweenOuterColumns();
//	TestFiltering_1D();
	//TestSimilarityWindowsCorrection();
	//TestRorateImage();
	//TestGradientMean(pInImage, iWidth, iHeight);
	TestRotateAddImage(argc, argv);
	//	TestInterpolateBetweenFilterPoints();

	//TestSortingAlgorithm();
//	TestHorizontalMedianFilter_cpu(pInImage, iWidth, iHeight, 20);

/*	TestHorizontalMedianFilter_IPP(pInImage, iWidth, iHeight);
	TestHorizontalMedianFilterCuda(pInImage, iWidth, iHeight, 20);
	TestVerticalMedianFilterCuda(pInImage, iWidth, iHeight, 20);*/



	//delete[] pInImage;


	getchar();
	
}