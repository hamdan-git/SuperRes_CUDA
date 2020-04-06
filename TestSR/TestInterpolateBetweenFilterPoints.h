#pragma once
#include "Common.h"
#include<time.h>
#include <stdio.h>
#include <math.h>
#include <ipp.h>
#include <ippi.h>
#include <ipps.h>
#include <algorithm>
#include<vector>

//----------------------------------------------------------------------------------------------------------------------------------------------------------------
double InterpolateBetweenFilterPoints(float fImage, float fRef1, float fRef2, float fGain1, float fGain2, int iRefPointIndex, int iNumPoints)
{
	double fDiff = (double)(fRef2 - fRef1);// + 0.001;
	if (fDiff == 0.0) fDiff = 0.001; //small value

	double fBeta = (double)(fImage - fRef1) / fDiff;
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

	double fInterpolated = fAlpha * fGain1 + fBeta * fGain2;

	return fInterpolated;
}

//-------------------------------------------------------------------------
void InterpolateBetweenPoint_prep(float *pInData, float *pOutData, float *pRef1, float *pRef2, float *pGain1, float *pGain2, int iW, int iH, float fOneOver_s)
{
	int iFrameSize = iW * iH;
	for (long m = 0; m < iFrameSize; m++)
	{
		float fVal = (float)(pInData[m] * fOneOver_s);
		float fVal2 = (float)InterpolateBetweenFilterPoints(fVal, pRef1[m], pRef2[m], pGain1[m], pGain2[m], 0, 3 - 1);
		pOutData[m] += fVal2 * pInData[m];

	}
}



//--------------------------------------------------------------
void TestInterpoltaeBetweenFilterPoints_cpu(float *pData, float *pRef1, float *pRef2, float *pGain1, float *pGain2, int iW, int iH, float fOneOver_s)
{
	int iOutWidth = iW;
	int iOutHeight = iH;



	long iOutFrameSize = iOutWidth * iOutHeight;

	float *pOutData = new float[iOutFrameSize];


	memset(pOutData, 0, sizeof(float)*iOutFrameSize);

	//addWithCuda();

	clock_t start = clock();
	for (int i = 0; i < iIterations; i++)
	{
		InterpolateBetweenPoint_prep(pData, pOutData, pRef1, pRef2, pGain1, pGain2, iW, iH, 0.45);
	}

	printf("Interpolate between points cpu took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("c:\\Temp\\BetweenPoints_cpu.raw", pOutData, iOutWidth, iOutHeight);

	delete[] pOutData;
}

//--------------------------------------------------------------
void TestInterpoltaeBetweenFilterPoints_cpu2(float *pData, int iWidth, int iHeight, int iNumFrames, std::vector<float*> *obGain_TE, int iNumPoints, int iNumBins, float fps)
{
	int iFrameSize = iWidth * iHeight;
	float *pTempData = new float[iFrameSize];
	float *pOutData = new float[iFrameSize*iNumFrames];
	float fOneOver_s = 100.0f;// fps;// 1000000.0 / fAcqTime_us;

	clock_t start = clock();
	for (int iZ = 0; iZ < iNumFrames; iZ++)
	{
		double t = (double)(iZ)  * fps;
		//iBinIndex= (int)(std::ceil(t/binSize));
		int iBinIndex = (int)t;// (int)((float)t / (float)binSize);
		iBinIndex = iBinIndex < 0 ? 0 : iBinIndex;//  max(iBinIndex, 0);
		iBinIndex = iBinIndex > iNumBins - 1 ? iNumBins - 1 : iBinIndex;// min(iBinIndex, iNumBins - 1);

		float *pRefInData = &pData[iZ*iFrameSize];
		float *pRefOutData = &pOutData[iZ*iFrameSize];
		memset((float *)pTempData, 0, sizeof(float)*iFrameSize);
		for (int k = 0; k < iNumPoints - 1; k++)
		{
			float *pRefImageData1 = &obGain_TE->at(k)[iBinIndex*iFrameSize];
			float *pRefImageData2 = &obGain_TE->at(k+1)[iBinIndex*iFrameSize];
			float *pGainData1 = pRefImageData1 + (iFrameSize*iNumBins); //the second block are the gain while the first block is the ref images
			float *pGainData2 = pRefImageData2 + (iFrameSize*iNumBins); //the second part of beta value



			for (long m = 0; m < iFrameSize; m++)
			{
				float fVal = (float)(pRefInData[m] * fOneOver_s);
				float fVal2 = (float)InterpolateBetweenFilterPoints(fVal, pRefImageData1[m], pRefImageData2[m], pGainData1[m], pGainData2[m], k, iNumPoints - 1);
				pTempData[m] += fVal2 * pRefInData[m];
			}
		}
		memcpy(pRefOutData, pTempData, sizeof(float)*iFrameSize);
	}

	printf("Interpolate between points cpu took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);
	WriteRawData("c:\\Temp\\BetweenPoints_cpu.raw", pOutData, iWidth, iHeight, iNumFrames);

	delete[] pOutData;
	delete[] pTempData;
}

//--------------------------------------------------------------
void TestInterpoltaeBetweenFilterPoints_gpu(float *pData, int iWidth, int iHeight, int iNumFrames, std::vector<float*> *obGain_TE, int iNumPoints, int iNumBins, float fps)
{
	int iFrameSize = iWidth * iHeight;
	float *pTempData = new float[iFrameSize];
	float *pOutData = new float[iFrameSize*iNumFrames];
	float fOneOver_s = fps;// 1000000.0 / fAcqTime_us;

	addWithCuda();

	clock_t start = clock();
	InterpolateBetweenFilterPoints_Cuda(pData, pOutData, iWidth, iHeight, iNumFrames, obGain_TE, iNumPoints, iNumBins, fOneOver_s);

	printf("Interpolate between points gpu took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);
	WriteRawData("c:\\Temp\\BetweenPoints_gpu.raw", pOutData, iWidth, iHeight, iNumFrames);

	delete[] pOutData;

}




//--------------------------
void TestInterpolateBetweenFilterPoints()
{
	int iWidth = 768;// 1024;
	int iHeight = 512;// 256;
	int iNumFrames = 400;// 400;
	long iFrameSize = iWidth * iHeight;

	float *pRefImage = ReadRawData<float>("E:\\Images\\Customers\\RefGain\\RefGain768X512_3points_10bins.raw", iWidth, iHeight, 60);
	float *pInImage = ReadRawData<float>("E:\\Images\\Customers\\RefGain\\Input_768X512X400.raw", iWidth, iHeight, iNumFrames);
//	float *pInImage = ReadRawData("E:\\Images\\Customers\\RefGain\\Input_TE.raw", iWidth, iHeight, iNumFrames);
	if (pInImage == NULL) { printf("null input"); exit(0); }
	int iNumPoints = 3;
	int iNumBins = 10;
	std::vector<float *> pGain(iNumPoints);
	for (int k = 0; k < iNumPoints; k++)
	{
		pGain[k] = &pRefImage[k*(iNumBins*iFrameSize * 2)];
	}

	//TestRotate_cpu(pInImage, iWidth, iHeight, fTheta);
	//TestRotate_gpu(pInImage, iWidth, iHeight, fTheta);
	TestInterpoltaeBetweenFilterPoints_cpu2(pInImage, iWidth, iHeight, iNumFrames, &pGain, iNumPoints, iNumBins, 1.0f/40.0f);
	TestInterpoltaeBetweenFilterPoints_gpu(pInImage, iWidth, iHeight, iNumFrames, &pGain, iNumPoints, iNumBins, 1.0f/40.0f);



	delete[] pInImage;
	delete[] pRefImage;
}