#pragma once
#include "Common.h"
#include<time.h>
#include <stdio.h>
#include <math.h>
#include <ipp.h>
#include <ippi.h>
#include <ipps.h>
#include <algorithm>

//----------------------------------------------------------------------------
void FindDimentionAfterRotation(int iInWidth, int iInHeight, double theta, int &iOutWidth, int &iOutHeight)
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

//-----------------------------------------------------------------------------------------
IppStatus warpAffine(float* pSrc, IppiSize srcSize, int srcStep, float* pDst, IppiSize dstSize, int dstStep, const double coeffs[2][3])
{
	/* IPP functions status */
	IppStatus status = ippStsNoErr;

	/* number of image channels */
	const Ipp32u numChannels = 1;

	/* border value to extend the source image */
	Ipp64f pBorderValue[numChannels];

	/* sizes for WarpAffine data structure, initialization buffer, work buffer */
	int specSize = 0, initSize = 0, bufSize = 0;

	/* pointer to work buffer */
	Ipp8u* pBuffer = NULL;

	/* pointer to WarpAffine data structure */
	IppiWarpSpec* pSpec = NULL;

	/* set offset of the processing destination ROI */
	IppiPoint dstOffset = { 0, 0 };

	/* border type for affine transform */
	IppiBorderType borderType = ippBorderConst;

	/* direction of warp affine transform */
	IppiWarpDirection direction = ippWarpForward;

	/* set border value to extend the source image */
	for (int i = 0; i < numChannels; ++i) pBorderValue[i] = 0.0;

	/* computed buffer sizes for warp affine data structure and initialization buffer */
	status = ippiWarpAffineGetSize(srcSize, dstSize, ipp32f, coeffs, ippLinear, direction, borderType,
		&specSize, &initSize);

	/* allocate memory */
	pSpec = (IppiWarpSpec*)ippsMalloc_8u(specSize);

	/* initialize data for affine transform */
	if (status >= ippStsNoErr)
		status = ippiWarpAffineLinearInit(srcSize, dstSize, ipp32f, coeffs, direction, numChannels, borderType, pBorderValue, 0, pSpec);

	/* get work buffer size */
	if (status >= ippStsNoErr)
		status = ippiWarpGetBufferSize(pSpec, dstSize, &bufSize);

	/* allocate memory for work buffer */
	pBuffer = ippsMalloc_8u(bufSize);

	/* affine transform processing */
	//if (status >= ippStsNoErr) status = ippiWarpAffineLinear_8u_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
	if (status >= ippStsNoErr)
		status = ippiWarpAffineLinear_32f_C1R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, dstOffset, dstSize, (const IppiWarpSpec*)pSpec, pBuffer);

	/* free allocated memory */
	ippsFree(pSpec);
	ippsFree(pBuffer);

	return status;
}

//-----------------------------
IppiWarpSpec *get_warpAffine(float* pSrc, IppiSize srcSize, int srcStep, float* pDst, IppiSize dstSize, int dstStep, const double coeffs[2][3])
{
	/* IPP functions status */
	IppStatus status = ippStsNoErr;

	/* number of image channels */
	const Ipp32u numChannels = 1;

	/* border value to extend the source image */
	Ipp64f pBorderValue[numChannels];

	/* sizes for WarpAffine data structure, initialization buffer, work buffer */
	int specSize = 0, initSize = 0, bufSize = 0;

	/* pointer to work buffer */
 //   Ipp8u* pBuffer  = NULL;

	/* pointer to WarpAffine data structure */
	//IppiWarpSpec* pSpec = NULL;

	/* set offset of the processing destination ROI */
	IppiPoint dstOffset = { 0, 0 };

	/* border type for affine transform */
	IppiBorderType borderType = ippBorderConst;

	/* direction of warp affine transform */
	IppiWarpDirection direction = ippWarpForward;

	/* set border value to extend the source image */
	for (int i = 0; i < numChannels; ++i) pBorderValue[i] = 0.0;

	/* computed buffer sizes for warp affine data structure and initialization buffer */
	status = ippiWarpAffineGetSize(srcSize, dstSize, ipp32f, coeffs, ippLinear, direction, borderType,
		&specSize, &initSize);

	/* allocate memory */
	IppiWarpSpec *pSpec = (IppiWarpSpec*)ippsMalloc_8u(specSize);

	/* initialize data for affine transform */
	if (status >= ippStsNoErr)
		status = ippiWarpAffineLinearInit(srcSize, dstSize, ipp32f, coeffs, direction, numChannels, borderType, pBorderValue, 0, pSpec);

	/* get work buffer size */
	if (status >= ippStsNoErr)
		status = ippiWarpGetBufferSize(pSpec, dstSize, &bufSize);

	return pSpec;
	//pSpec_0 = pSpec;
	/* allocate memory for work buffer */
 //   pBuffer = ippsMalloc_8u(bufSize);

	/* affine transform processing */
	//if (status >= ippStsNoErr) status = ippiWarpAffineLinear_8u_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
//    if (status >= ippStsNoErr) 
//		status = ippiWarpAffineLinear_32f_C1R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, dstOffset, dstSize, (const IppiWarpSpec*)pSpec, pBuffer);

	/* free allocated memory */
//    ippsFree(pSpec);
//    ippsFree(pBuffer);

	//return status;
}
//-----------------------------------------------------------------
IppStatus RotateImage_ipp(float *pInData, int iSrcWidth, int iSrcHeight, float *pOutData, int iDestWidth, int iDestHeight, double fAngle)
{
	IppiSize roiSize = { iDestWidth / 1, iDestHeight };
	IppiRect srcRoi = { 0, 0, iSrcWidth, iSrcHeight };

	/* affine transform coefficients */
	double coeffs[2][3] = { 0 };

	/* affine transform bounds */
	double bound[2][2] = { 0 };

	IppStatus status = ippiGetRotateTransform(fAngle, 0, 0, coeffs);

	/* get bounds of transformed image */
	if (status >= ippStsNoErr)
		status = ippiGetAffineBound(srcRoi, bound, coeffs);


	/* fit source image to dst */
	coeffs[0][2] = -bound[0][0] + (iDestWidth / 1.f - (bound[1][0] - bound[0][0])) / 1.f;
	coeffs[1][2] = -bound[0][1] + (iDestHeight / 1 - (bound[1][1] - bound[0][1])) / 1.f;

	/* set destination ROI for the not blurred image */
	//pDstRoi = pBlurRot + roiSize.width * numChannels;
	IppiSize srcSize = { iSrcWidth, iSrcHeight };
	if (status >= ippStsNoErr)
		status = warpAffine(pInData, srcSize, iSrcWidth * 4, pOutData, roiSize, iDestWidth * 4, coeffs);

	return status;
}

//----------------------------------------------------------------------
/** Uses bilinear interpolation to find the pixel value at real coordinates (x,y). */
template<typename T>
double getInterpolatedPixel2(double x, double y, int iWidth, int iHeight, T* pixels)
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
		//return (lowerLeft + lowerRight + upperRight + upperLeft) / 4.0;
		double fSum = 0.0;
		int count = 0;
		int win = 1;
		for (int j = -win; j <= win; j++)
		{
			for (int i = -win; i <= win; i++)
			{
				int newX = xbase + i;
				int newY = ybase + j;
				if (newX >= 0 && newX < iWidth && newY >= 0 && newY < iHeight)
				{
					count++;
					fSum += pixels[newY*iWidth  + newX];
				}
			}
		}
		if (count > 0)
			fSum = fSum / (double)8;
		return fSum;
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

//----------------------------------------------------------------------
/** Uses bilinear interpolation to find the pixel value at real coordinates (x,y). */
template<typename T>
double getInterpolatedPixel(double x, double y, int iWidth, int iHeight, T* pixels)
{
	int xbase = (int)x;
	int ybase = (int)y;
	if (xbase < -1 || ybase < -1 || xbase > iWidth+2 || ybase > iHeight+2) 
		return 0.0;
	xbase = min(xbase, iWidth - 1);
	ybase = min(ybase, iHeight - 1);
	xbase = max(xbase, 0);
	ybase = max(ybase, 0);
	//if (xbase  >= iWidth  ybase >= iHeight )
	//	return 1;
	double xFraction = x - xbase;
	double yFraction = y - ybase;
	int offset = ybase * iWidth + xbase;
	double lowerLeft = pixels[offset];
	double lowerRight = xbase==iWidth-1 ? pixels[offset] : pixels[offset + 1];
	double upperRight = (xbase==iWidth-1||ybase==iHeight-1) ? pixels[offset]: pixels[offset + iWidth + 1];
	double upperLeft = ybase==iHeight-1 ? pixels[offset]:pixels[offset + iWidth];
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

//--------------------------------------------------------------------
void RotateImage_cpu(float *pData, int iW, int iH, float *pOutData, int iOutWidth, int iOutHeight, double theta)
{
	memset(pOutData, 0, 4 * iOutWidth*iOutHeight);
	float rads = (theta) * 3.1415926 / 180.0;
	float cs = cos(rads); // precalculate these values
	float ss = sin(rads);
	float xcenterOut = (float)(iOutWidth) / 2.0;   // use float here!
	float ycenterOut = (float)(iOutHeight) / 2.0;
	float xcenterIn = (float)iW / 2.0f;
	float ycenterIn = (float)iH / 2.0f;
	for (int row = 0; row < iOutHeight; row++)
	{
		for (int col = 0; col < iOutWidth; col++)
		{
			float u = (float)col - xcenterOut;
			float v = (float)row - ycenterOut;
			float tu = u * cs - v * ss;
			float tv = v * cs + u * ss;

			tu += xcenterIn;
			tv += ycenterIn;
			//tu += (iOutWidth - iW) / 2;
			//tu += (iOutHeight - iH) / 2;

			//if (tu >= 0 && tu < iW && tv >= 0 && tv < iH)
			{
			//	pOutData[row*iOutWidth + col] = pData[(int)tv*iW + (int)tu];
				pOutData[row*iOutWidth + col] = getInterpolatedPixel2(tu, tv, iW, iH,  pData);
			}
		}
	}
}


//--------------------------------------------------------------
void TestRotate_cpu(float *pData, int iW, int iH, double theta)
{

	int iOutWidth = iW;
	int iOutHeight = iH;

	FindDimentionAfterRotation(iW, iH, theta, iOutWidth, iOutHeight);

	int iOutFrameSize = iOutWidth * iOutHeight;

	float *pOutData = new float[iOutFrameSize];

	clock_t start = clock();
	for (int i = 0; i < iIterations; i++)
	{
		//	IppStatus st = RotateImage_ipp(pData, iW, iH, pOutData, iOutWidth, iOutHeight, theta);
		RotateImage_cpu(pData, iW, iH, pOutData, iOutWidth, iOutHeight, theta);
	}

	printf("Rotate cpu (ipp) took average of %f s  dim=(%d, %d)\n", ((double)(clock() - start) / (double)CLOCKS_PER_SEC) / iIterations, iOutWidth, iOutHeight);

	WriteRawData("c:\\Temp\\Rotate_cpu.raw", pOutData, iOutWidth, iOutHeight);

	delete[] pOutData;

}
//--------------------------------------------------------------
void TestRotate_ipp_cpu(float *pData, int iW, int iH, double theta)
{

	int iOutWidth = iW;
	int iOutHeight = iH;

	FindDimentionAfterRotation(iW, iH, theta, iOutWidth, iOutHeight);

	int iOutFrameSize = iOutWidth * iOutHeight;

	float *pOutData = new float[iOutFrameSize];

	clock_t start = clock();
	for (int i = 0; i < iIterations; i++)
	{
		IppStatus st = RotateImage_ipp(pData, iW, iH, pOutData, iOutWidth, iOutHeight, theta);
		//RotateImage_cpu(pData, iW, iH, pOutData, iOutWidth, iOutHeight, theta);
	}

	printf("Rotate cpu (ipp) took average of %f s  dim=(%d, %d)\n", ((double)(clock() - start) / (double)CLOCKS_PER_SEC)/ iIterations, iOutWidth, iOutHeight);

	WriteRawData("c:\\Temp\\Rotate_ipp_cpu.raw", pOutData, iOutWidth, iOutHeight);

	delete[] pOutData;
}


//--------------------------------------------------------------
void TestRotate_gpu(float *pData, int iW, int iH, double theta)
{
	int iOutWidth = iW;
	int iOutHeight = iH;

	FindDimentionAfterRotation(iW, iH, theta, iOutWidth, iOutHeight);

	int iOutFrameSize = iOutWidth * iOutHeight;

	float *pOutData = new float[iOutFrameSize];

	memset(pOutData, 0, sizeof(float)*iOutFrameSize);

	addWithCuda();

	clock_t start = clock();
	for (int i = 0; i < iIterations; i++)
	{
		RotateImage_tex_Cuda(pData, iW, iH, pOutData, iOutWidth, iOutHeight, theta);
	}

	printf("Rotate gpu (tex) took average of %f s\n", ((double)(clock() - start) / (double)CLOCKS_PER_SEC)/ iIterations);

	WriteRawData("c:\\Temp\\RotateImage_gpu.raw", pOutData, iOutWidth, iOutHeight);

	delete[] pOutData;
}

//cudaError_t Filtering_1D_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, float *pKernel, int iKernelSize, int dir)

//--------------------------
void TestRorateImage()
{
	int iWidth = 2048;// 1024;
	int iHeight = 64;// 256;
	double fTheta = -15.5;


	float *pInImage = ReadRawData<float>("C:\\Temp\\TestSR.raw", iWidth, iHeight);
	if (pInImage == NULL) { printf("null input"); exit(0); }



	TestRotate_cpu(pInImage, iWidth, iHeight, fTheta);
	TestRotate_ipp_cpu(pInImage, iWidth, iHeight, fTheta);
	TestRotate_gpu(pInImage, iWidth, iHeight, fTheta);



	delete[] pInImage;
}