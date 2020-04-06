#pragma once
#include "Common.h"
#include<time.h>
#include <stdio.h>
#include <math.h>
#include <ipp.h>
#include <ippi.h>
#include <ipps.h>
#include <algorithm>
#include "IPTools.h"

//----------------------------------------------------------------------------
void FindDimentionAfterRotationAdd(int iInWidth, int iInHeight, double theta, int &iOutWidth, int &iOutHeight)
{
	// Compute dimensions of the resulting bitmap
	// First get the coordinates of the 3 corners other than origin
	double rads = theta * 3.1415926 / 180.0; // fixed constant PI
	double cs = cos(-rads); // precalculate these values
	double ss = sin(-rads);
	int x1 = (int)(iInHeight * ss);
	int y1 = (int)(iInHeight * cs);
	int x2 = (int)(iInWidth * cs + iInHeight * ss);
	int y2 = (int)(iInHeight * cs - iInWidth * ss);
	int x3 = (int)(iInWidth * cs);
	int y3 = (int)(-iInWidth * ss);

	int minx = min(0, min(x1, min(x2, x3)));
	int miny = min(0, min(y1, min(y2, y3)));
	int maxx = max(0, max(x1, max(x2, x3)));
	int maxy = max(0, max(y1, max(y2, y3)));

	iOutWidth = maxx - minx;
	iOutHeight = maxy - miny;
}

//-----------------------------------------------------------------
//IppStatus RotateImage_ipp(float *pInData, int iSrcWidth, int iSrcHeight, float *pOutData, int iDestWidth, int iDestHeight, double fAngle)
//{
//	IppiSize roiSize = { iDestWidth / 1, iDestHeight };
//	IppiRect srcRoi = { 0, 0, iSrcWidth, iSrcHeight };
//
//	/* affine transform coefficients */
//	double coeffs[2][3] = { 0 };
//
//	/* affine transform bounds */
//	double bound[2][2] = { 0 };
//
//	IppStatus status = ippiGetRotateTransform(fAngle, 0, 0, coeffs);
//
//	/* get bounds of transformed image */
//	if (status >= ippStsNoErr)
//		status = ippiGetAffineBound(srcRoi, bound, coeffs);
//
//
//	/* fit source image to dst */
//	coeffs[0][2] = -bound[0][0] + (iDestWidth / 1.f - (bound[1][0] - bound[0][0])) / 1.f;
//	coeffs[1][2] = -bound[0][1] + (iDestHeight / 1 - (bound[1][1] - bound[0][1])) / 1.f;
//
//	/* set destination ROI for the not blurred image */
//	//pDstRoi = pBlurRot + roiSize.width * numChannels;
//	IppiSize srcSize = { iSrcWidth, iSrcHeight };
//	if (status >= ippStsNoErr)
//		status = warpAffine(pInData, srcSize, iSrcWidth * 4, pOutData, roiSize, iDestWidth * 4, coeffs);
//
//	return status;
//}

//--------------------------------------------------------------
//void TestRotate_cpu(float *pData, int iW, int iH, double theta)
//{
//
//	int iOutWidth = iW;
//	int iOutHeight = iH;
//
//	FindDimentionAfterRotation(iW, iH, theta, iOutWidth, iOutHeight);
//
//	int iOutFrameSize = iOutWidth * iOutHeight;
//
//	float *pOutData = new float[iOutFrameSize];
//
//	clock_t start = clock();
//	for (int i = 0; i < iIterations; i++)
//	{
//		IppStatus st = RotateImage_ipp(pData, iW, iH, pOutData, iOutWidth, iOutHeight, theta);
//	}
//
//	printf("Rotate cpu (ipp) took average of %f s  dim=(%d, %d)\n", ((double)(clock() - start) / (double)CLOCKS_PER_SEC) / iIterations, iOutWidth, iOutHeight);
//
//	WriteRawData("c:\\Temp\\Rotate_ipp_cpu.raw", pOutData, iOutWidth, iOutHeight);
//
//	delete[] pOutData;
//}

//----------------------------------------------------------------------------
//unsigned char *GetMaskDataAfterRotation(int iW, int iH, double theta, int &iNewW, int &iNewH )
//{
//	unsigned char *pMaskData = NULL;
//	//int iNewW=0, iNewH = 0;
//	FindDimensionAfterRotation(iW, iH, theta, iNewW, iNewH);
//	if (iNewW <= 0 || iNewH <= 0) return NULL;
//	pMaskData = new unsigned char[iNewW*iNewH];
//	unsigned char *pTempData = new unsigned char[iW*iH];
//	unsigned char *pRotatedTempData = new unsigned char[iNewW*iNewH];
//	memset(pRotatedTempData, 0, iNewW*iNewH);
//
//	for (long i = 0; i < iW*iH; i++)
//		pTempData[i] = 1;
//
//	unsigned char *pMaks_rot_eroded = new unsigned char[iNewW*iNewH];
//	memset(pMaks_rot_eroded, 0, iNewW*iNewH);
//	IPTools<unsigned char>::RotateImage_cpu(pTempData, iW, iH, pRotatedTempData, iNewW, iNewH, theta, 0);
//	IPTools<unsigned char>::DoErosion(pRotatedTempData, pMaks_rot_eroded, iNewW, iNewH, 3);
//	float *pDisData = new float[iNewW*iNewH];
//	IPTools<unsigned char>::GetDistanceMap(pMaks_rot_eroded, iNewW, iNewH, 0, pDisData);
//
//	//WriteRawData<unsigned char>("c:\\Temp\\Dist.raw", pMaks_rot_eroded, iNewW, iNewH);
//
//	memset(pMaskData, 0, 1 * iNewW*iNewH);
//	for (long i = 0; i < iNewW*iNewH; i++)
//	{
//		float fVal = pDisData[i];
//		if (fVal > 0.0f && fVal < 25.0f)
//			pMaskData[i] = 1;
//	}
//
//	delete[] pTempData;
//	delete[] pRotatedTempData;
//	delete[] pMaks_rot_eroded;
//	delete[] pDisData;
//	return pMaskData;
//}

//--------------------------------------------------------------
void TestRotateAdd_gpu(unsigned short *pData, int iW, int iH, int iD, double theta, float fShiftStep)
{

	int iRotWidth = iW;
	int iRotHeight = iH;

	FindDimentionAfterRotationAdd(iW, iH, theta, iRotWidth, iRotHeight);

	int iOutWidth = iRotWidth;
	int iOutHeight = iRotHeight + iD * fShiftStep + 2;;


	int iOutFrameSize = iOutWidth * iOutHeight;

	unsigned short *pOutData = new unsigned short[iOutFrameSize];

	memset(pOutData, 0, sizeof(unsigned short)*iOutFrameSize);

	addWithCuda();


	clock_t start = clock();

	//Get mask data to cover arround the edges after rotation
	//int iNewMaskW = 0, iNewMaskH = 0;
	//unsigned char *pMaskData = GetMaskDataAfterRotation(iW, iH, theta, iNewMaskW , iNewMaskH);
	//if (iNewMaskW != iRotWidth || iNewMaskH != iRotHeight)
	//{
	//	printf("dimension mismatch\n");
	//	goto cleanup_TestRotateAdd_gpu;
	//}
	//WriteRawData<unsigned char>("c:\\Temp\\MaskData.raw", pMaskData, iNewMaskW, iNewMaskH);
	//for (int i = 0; i < iIterations; i++)
	{
		RotateAddImage_tex_Cuda(pData, iW, iH, iD, pOutData, iOutWidth, iOutHeight,  theta, fShiftStep);
	}

	printf("Rotate gpu (tex) took average of %f s  (%d, %d)\n", ((double)(clock() - start) / (double)CLOCKS_PER_SEC) , iOutWidth, iOutHeight);

	WriteRawData<unsigned short>("c:\\Temp\\RotateAddImage_gpu.raw", pOutData, iOutWidth, iOutHeight);

cleanup_TestRotateAdd_gpu:
	delete[] pOutData;
	//delete[] pMaskData;
}

//cudaError_t Filtering_1D_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, float *pKernel, int iKernelSize, int dir)

//--------------------------
void TestRorateAddImage()
{
	int iWidth = 2048;// 1024;
	int iHeight = 64;// 256;
	int iNumFrames = 2000;
	double fTheta = -15.5;
	double fShiftScale = 0.65;

	unsigned short *pInImage = ReadRawData<unsigned short>("D:\\Images\\XCounter\\Customers\\0_Internal\\Hamdan\\SuperRes\\Detector2\\Target\\frame_200hz_ang_-15.5_angled_target_gainCor_u16.raw", iWidth, iHeight, iNumFrames);
	if (pInImage == NULL) { printf("null input"); exit(0); }


	//TestRotate_cpu(pInImage, iWidth, iHeight, fTheta);
	TestRotateAdd_gpu(pInImage, iWidth, iHeight, iNumFrames, fTheta, fShiftScale);



	delete[] pInImage;
}