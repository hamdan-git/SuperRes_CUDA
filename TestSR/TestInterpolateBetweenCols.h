#pragma once
#include "Common.h"


//---------------------------------------------------------------------------------------------
int GetYPosGivenSlope_MCP(int iRefX, float fSlope, int iCurX, int iCurY)
{
	return (int)(fSlope*(float)(iRefX - iCurX)) + iCurY;
}

//-----------------------------------------------------------------------------------------------
float GetDistanceBetweenTwoPoints_MCP(int x1, int y1, int x2, int y2)
{
	return (float)sqrt((double)((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)));
}

//----------------------------------------------------------------------------------
float InterpolatePixelFrom2Points_MCP(float fVal1, float fDist1, float fVal2, float fDist2)
{
	float fDenom = fDist1 + fDist2;
	if (fDenom != 0.0f)
	{
		fDenom = 1.0f / fDenom;
		return (fVal1*fDist2 + fVal2 * fDist1)*fDenom;
	}
	return 0.0;
}


//---------------------------------------------------------------------------------------------------
template <typename T>
bool InterpolateBetweenOuterColumns_MTS_MCP(const T *pInImage, T *pOutImage, int iWidth, int iHeight, int iStartX, int iEndX)
{
	try
	{
		if (iEndX >= iWidth || iStartX < 0) return false;

		memcpy(pOutImage, pInImage, sizeof(T) * iHeight*iWidth);
		float fCureVal = 0.0;
		float fSlopes[3] = { -1.0f, 0.0f, 1.0f }; //3 directions to search for best fit
		float fVals[3]; //containg the interpolated values in different directions
		float fGrad[3]; //containing gradient in different directions
		int iY1 = 0, iY2 = 0; //y index given the slope
		float fVal1 = 0.0f, fVal2 = 0.0f; //values of new pixels in the direction of the given slope 
		float fDist1, fDist2; //distances between new and current point to see find the contribution of the new points
		int iMinIndex = -1; //which direction has the lowest gradient
		float fMinGrad = 0.0; //to keep track of min gradient 
		float fCurGrad = 0.0;
		//use the outer columsn as reference and interpolate the central columns
		for (int iX = iStartX + 1; iX <= iEndX - 1; iX++)
		{
			long iOffsetY = 0;
			for (int iY = 0; iY < iHeight; iY++)
			{
				fCureVal = pInImage[iOffsetY + iX];
				//go to different directions
				fMinGrad = FLT_MAX;
				iMinIndex = -1;
				for (int k = 0; k < 3; k++)
				{
					iY1 = GetYPosGivenSlope_MCP(iStartX, fSlopes[k], iX, iY);
					iY1 = max(min(iY1, iHeight - 1), 0);
					iY2 = GetYPosGivenSlope_MCP(iEndX, fSlopes[k], iX, iY);
					iY2 = max(min(iY2, iHeight - 1), 0);
					fDist1 = GetDistanceBetweenTwoPoints_MCP(iStartX, iY1, iX, iY);
					fDist2 = GetDistanceBetweenTwoPoints_MCP(iEndX, iY2, iX, iY);
					//get the pixel values
					fVal1 = pInImage[iY1*iWidth + iStartX];
					fVal2 = pInImage[iY2*iWidth + iEndX];

					//find the interpolated values between the two pixels using distance as a contributing factor
					fVals[k] = InterpolatePixelFrom2Points_MCP(fVal1, fDist1, fVal2, fDist2);
					fCurGrad = (float)fabs(fVal1 - fVal2);
					fGrad[k] = fCurGrad;
					if (fCurGrad < fMinGrad)
					{
						fMinGrad = fCurGrad; iMinIndex = k;
					}
					//iMinIndex = 1;
				}

				if (iMinIndex >= 0)
				{
					pOutImage[iOffsetY + iX] = fVals[iMinIndex];
				}
				iOffsetY += iWidth;
			}
		}

	}
	catch (...)
	{
	}
	return true;
}

//--------------------------------------------------------------
void TestInterpolateBetween_cpu(float *pData, int iW, int iH)
{
	int iFrameSize = iW * iH;
	float *pOutData = new float[iFrameSize];

	clock_t start = clock();
	for (int i = 0; i < iIterations; i++)
	{
		InterpolateBetweenOuterColumns_MTS_MCP(pData, pOutData, iW, iH, 0, 100);
	}

	printf("Interpolate cpu took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("E:\\Temp\\Interpolate_cpu.raw", pOutData, iW, iH);

	delete[] pOutData;
}


//--------------------------------------------------------------
void TestInterpolateBetween_gpu(float *pData, int iW, int iH)
{
	int iFrameSize = iW * iH;
	float *pOutData = new float[iFrameSize];

	addWithCuda();

	clock_t start = clock();
	for (int i = 0; i < iIterations; i++)
	{
		interpolateBetweenColumnsCuda(pData, pOutData, iW, iH, 0, 100);
	}

	printf("Interpolate gpu took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("E:\\Temp\\Interpolate_gpu.raw", pOutData, iW, iH);

	delete[] pOutData;
}


//--------------------------
void TestInterpolateBetweenOuterColumns()
{
	int iWidth = 1024;// 1024;
	int iHeight = 1024;// 256;
	float *pInImage = ReadRawData<float>("E:\\Temp\\TestMedian.raw", iWidth, iHeight);
	if (pInImage == NULL) { printf("null input"); exit(0); }

	TestInterpolateBetween_cpu(pInImage, iWidth, iHeight);
	TestInterpolateBetween_gpu(pInImage, iWidth, iHeight);

}