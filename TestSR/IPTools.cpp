#include "IPTools.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <algorithm>    // std::min
using namespace std;

//----------------------------------------------------------------------------
void FindDimensionAfterRotation(int iInWidth, int iInHeight, double theta, double fMag, int &iOutWidth, int &iOutHeight)
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

	iOutWidth *= fMag;
	iOutHeight *= fMag;
}
//--------------------------------------------------------------
template<typename T>
void IPTools<T>::DoErosion(const T *pInData, T* pOutData, int iWidth, int iHeight, int iWin)
{

	int iStrElemWidth = iWin;
	int iStrElemHeight = iWin;

	int* piStrElemMatrix = 0;
	int iStrElemSize = iStrElemWidth * iStrElemHeight;
	try
	{
		if (iWin <= 3)
		{
			piStrElemMatrix = new int[iStrElemSize];

			for (int k = 0; k < iStrElemSize; k++)
			{
				piStrElemMatrix[k] = 1;
			}
		}

		if (iWin > 3)
		{
			int iRad = int(iWin / 2);

			iStrElemWidth = 2 * iRad + 1;
			iStrElemHeight = 2 * iRad + 1;

			iStrElemSize = iStrElemWidth * iStrElemHeight;

			piStrElemMatrix = new int[iStrElemSize];

			for (int i = 0; i < iStrElemSize; i++)
				piStrElemMatrix[i] = 0;

			int nRadSq = iRad * iRad;
			int nStart = -iRad;
			int nEnd = iRad;
			for (int j = nStart; j <= nEnd; j++)
			{
				for (int i = nStart; i <= nEnd; i++)
				{
					int nIndexPoint = (i + iRad) + (j + iRad) * iStrElemWidth;
					if ((i * i + j * j) <= nRadSq)
					{
						piStrElemMatrix[nIndexPoint] = 1;
					}
					else
					{
						piStrElemMatrix[nIndexPoint] = 0;
					}
				}
			}

			//MCircleElementStrategy circleST(iRad, piStrElemMatrix);
			//circleST.Execute();						
		}

		int iHalfMaskWidth = (iStrElemWidth - 1) / 2;
		int iHalfMaskHeight = (iStrElemHeight - 1) / 2;

		//const MBinaryImageType* pbOrigInputData = pTempInputDataCont->GetSliceData(0);
		//const MBinaryImageType *pbMask = pbOrigInputData;
		T * pbOrigInputData = (T*)pInData;
		T *pbMask = (T*)pInData;
		//MBinaryImageType* pbResData = pResultDataCont->GetSliceData(0);
		T *pbResData = pOutData;

		for (int j = 0; j < iHeight; j++)
		{
			int iTop = j - iHalfMaskHeight;
			int iBottom = j + iHalfMaskHeight;

			if (iTop < 0) iTop = 0;
			if (iBottom >= iHeight) iBottom = iHeight - 1;

			int iTopOffset = iTop - (j - iHalfMaskHeight);

			for (int i = 0; i < iWidth; i++)
			{
				if (*pbMask > 0)
				{
					int iLeft = i - iHalfMaskWidth;
					int iRight = i + iHalfMaskWidth;

					if (iLeft < 0) iLeft = 0;
					if (iRight >= iWidth) iRight = iWidth - 1;

					int iLeftOffset = iLeft - (i - iHalfMaskWidth);
					int iRightOffset = (i + iHalfMaskWidth) - iRight;

					int iToWidth = iWidth - 1 - iRight;

					//const MBinaryImageType* pTempInputData = pbOrigInputData + iTop * iWidth;
					const T* pTempInputData = pbOrigInputData + iTop * iWidth;

					int iCounter = iTopOffset * iWin;

					bool bIsValid = true;

					for (int n = iTop; n <= iBottom; n++)
					{
						pTempInputData += iLeft;
						iCounter += iLeftOffset;

						for (int m = iLeft; m <= iRight; m++)
						{
							if (piStrElemMatrix[iCounter] && !*pTempInputData)
							{
								bIsValid = false;
								break;
							}
							pTempInputData++;
							iCounter++;
						}
						if (!bIsValid)
							break;

						pTempInputData += iToWidth;
						iCounter += iRightOffset;
					}
					if (bIsValid)
						*pbResData = 127;
					else
						*pbResData = 0;
				}
				pbMask++;
				pbResData++;
			}
		}

		delete[] piStrElemMatrix;
	}
	catch (short a)
	{
		if (piStrElemMatrix)
			delete[] piStrElemMatrix;
	}
}


//-------------------------------------------
template<typename T>
void IPTools<T>::GetDistanceMap(const T* pInData, int iWidth, int iHeight, T *pOutData = 0, float *pFloatData = 0)
{

	int iDepth = 1; //2D for now

	long iFrameSize = iWidth * iHeight;

	float fResX = 1.0;//m_pbinaryimgInput->GetResX();
	float fResY = 1.0;//m_pbinaryimgInput->GetResY();

	float *pTempOutData = new float[iFrameSize];
	memset(pTempOutData, 0, sizeof(float)*iFrameSize);
	float *pDistCmp = new float[iFrameSize];
	memset(pDistCmp, 0, sizeof(float)*iFrameSize);
	//TImage<float> TempOutImage;
	//TempOutImage.Build(iWidth, iHeight, 1);
	//TImage<float> imgDistComparison;  //temporal image used to compare if the algorithm has converged
	//imgDistComparison.Build(iWidth, iHeight, 1);		
	//const MBinaryImageType* pbinInput = pInputDataCont->GetSliceData(0);  //get a pointer to  the input data
	const T* pbinInput = pInData;  //get a pointer to  the input data
	//float* pfOutput = pOutputDataCont->GetSliceData(0);   //a pointer to the output data
	float* pfOutput = pTempOutData;   //a pointer to the output data

	//floatDataContainer* pDistComparisonDataCont = (floatDataContainer*)imgDistComparison.GetDataContainer();	
	//float* pfComp = pDistComparisonDataCont->GetSliceData(0);
	float* pfComp = pDistCmp;//->GetSliceData(0);


	long iIterator = iFrameSize;//pInputDataCont->GetSliceDataSize();  //a loop for the scanning of all pixels
	bool bIsOnlyBackground = true;



	do
	{
		if (*pbinInput)
		{
			*pfComp = *pfOutput = 0.0f;   //pixels part of the binary object are set a zero distance
			bIsOnlyBackground = false;
		}
		else
			*pfComp = *pfOutput = 10000000.0f;	 //background pixels are given a very large value as distance

		pbinInput++;
		pfComp++;
		pfOutput++;

	} while (--iIterator);

	if (bIsOnlyBackground)
	{
		//pOutputDataCont->ResetData(0);
		//memset(TempOutImage.m_ppData[0], 0, sizeof(float)*iFrameSize);
		//m_pgreyimgOutput->CalculateRange();
		//return;
		goto Dist_Cleanup;
	}

	float ppfMask[5][5];

	float fDistX = fResX;
	float fDistY = fResY;
	float fDistX2 = fDistX * fDistX;
	float fDistY2 = fDistY * fDistY;

	float fDistXDistY = (float)sqrt(fDistX2 + fDistY2);
	float f2DistXDistY = (float)sqrt(4.0f*fDistX2 + fDistY2);
	float fDistX2DistY = (float)sqrt(fDistX2 + 4.0f*fDistY2);


	ppfMask[0][0] = 123456.0f;  //mask used for getting the distance of the image
	ppfMask[0][1] = fDistX2DistY;		//see the technical report of distance transform to see
	ppfMask[0][2] = 123456.0f;	//how this values were chosen
	ppfMask[0][3] = fDistX2DistY;
	ppfMask[0][4] = 123456.0f;

	ppfMask[1][0] = f2DistXDistY;
	ppfMask[1][1] = fDistXDistY;
	ppfMask[1][2] = fDistY;
	ppfMask[1][3] = fDistXDistY;
	ppfMask[1][4] = f2DistXDistY;

	ppfMask[2][0] = 123456.0f;
	ppfMask[2][1] = fDistX;
	ppfMask[2][2] = 0.0f;
	ppfMask[2][3] = fDistX;
	ppfMask[2][4] = 123456.0f;

	ppfMask[3][0] = f2DistXDistY;
	ppfMask[3][1] = fDistXDistY;
	ppfMask[3][2] = fDistY;
	ppfMask[3][3] = fDistXDistY;
	ppfMask[3][4] = f2DistXDistY;

	ppfMask[4][0] = 123456.0f;
	ppfMask[4][1] = fDistX2DistY;
	ppfMask[4][2] = 123456.0f;
	ppfMask[4][3] = fDistX2DistY;
	ppfMask[4][4] = 123456.0f;


	int iterNumber = 0;

	bool bHasConverged = true;
	do
	{
		float* pfOutput = pTempOutData;//pOutputDataCont->GetSliceData(0);
		iterNumber++;			//iteration number
		for (int i = 0; i < iHeight; i++)  //for all pixel locations starting from top-left to bottom-right
		{											//We could assume that the first and last two rows are columns
													//are zero, however if a non-background pixel is in this location
													//it would not be possible to examine it
			for (int j = 0; j < iWidth; j++)
			{
				float fMinimum = 1e10f; //a sufficiently large number is set as the minimum so that this
										 //value can be updated in the following loop
											//this value should be higher than the maximum value given in the mask
											//which is 10000000.0f or 1e7
				if (*pfOutput > 0)
				{
					bool bDone = false;
					for (int k = -2; k <= 0; k++) //only the upper half of the mask will be used for the forward
					{
						int iYOffset = (i + k)*iWidth;	//computation of the distance
						for (int l = -2; l <= 2; l++)
						{
							if (ppfMask[k + 2][l + 2] != 123456.0f)
							{
								if ((j + l) >= 0 && (j + l) < iWidth && (i + k) >= 0 && (i + k) < iHeight)
								{
									float fValue = pTempOutData[iYOffset + j + l] + ppfMask[k + 2][l + 2];
									if (fValue <= fMinimum)
										fMinimum = fValue;		//choosing minimum distance
								}
								if (k == 0 && l == 0)
								{
									bDone = true;
									break;
								}
							}
						}
						if (bDone)
							break;
					}
				}

				if (*pfOutput > fMinimum)
					*pfOutput = fMinimum;  //the minimum distance is set for each pixel
				pfOutput++;

			}
		}



		pfOutput = pTempOutData;//pOutputDataCont->GetSliceData(0);
		pfOutput += iWidth * iHeight - 1;  //positioning the pointer in the last location
		for (int i = iHeight - 1; i >= 0; i--) //for all pixel locations starting from bottom-right to top-left
		{
			for (int j = iWidth - 1; j >= 0; j--)
			{
				float fMinimum = 1e10f;
				bool bDone = false;
				if (*pfOutput > 0)
				{
					for (int k = 2; k >= 0; k--)  //only the lower half of the mask is used
					{
						int iYOffset = (i + k)*iWidth;
						for (int l = 2; l >= -2; l--)
						{
							if (ppfMask[k + 2][l + 2] != 123456.0f)
							{
								if ((j + l) >= 0 && (j + l) < iWidth && (i + k) >= 0 && (i + k) < iHeight)
								{
									float fValue = pTempOutData[iYOffset + j + l] + ppfMask[k + 2][l + 2];
									if (fValue <= fMinimum)
										fMinimum = fValue;
								}
								if (k == 0 && l == 0)
								{
									bDone = true;
									break;
								}
							}
						}
						if (bDone)
							break;
					}
				}
				if (*pfOutput > fMinimum)
					*pfOutput = fMinimum;     //taking the minimum
				pfOutput--;


			}
		}

		iIterator = iFrameSize;//pOutputDataCont->GetSliceDataSize();
		pfOutput = pTempOutData;//OutputDataCont->GetSliceData(0);
		float* pfComp = pDistCmp;//pDistComparisonDataCont->GetSliceData(0);
		bHasConverged = true;
		do
		{
			if (*pfComp != *pfOutput || *pfOutput == 123456)
			{
				bHasConverged = false;
				break;
			}

			pfComp++;
			pfOutput++;
		} while (--iIterator);

		//copying the new distance image to the original data for next iteration			
		//pDistComparisonDataCont->InitialiseData(*pOutputDataCont);
		memcpy(pDistCmp, pTempOutData, sizeof(float)*iFrameSize);

	} while (!bHasConverged && iterNumber < 0);  //exit if the number of iterations has been reached

	//m_pgreyimgOutput->CalculateRange();

	if (pOutData)
	{
		T *pOutData = pOutData;
		float *pFloatOutData = pTempOutData;
		for (int iY = 0; iY < iHeight; iY++)
		{
			for (int iX = 0; iX < iWidth; iX++)
			{
				*pOutData = (T)*pFloatOutData;
				pOutData++;
				pFloatOutData++;
			}
		}
	}

	//if float image is requested
	if (pFloatData)
	{
		memcpy(pFloatData, pTempOutData, sizeof(float)*(iWidth*iHeight));
	}
	//IPTools<float>::WriteSMVImage("DistFloat1.smv", TempOutImage);
Dist_Cleanup:
	delete[] pTempOutData;
	delete[] pDistCmp;
}



//------------------------------------------------------------------
template<typename T>
double getInterpolatedPixel_TF(double x, double y, int iWidth, int iHeight, T* pixels)
{
	int xbase = (int)x;
	int ybase = (int)y;
	xbase = min(xbase, iWidth - 1);
	ybase = min(ybase, iHeight - 1);
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


//--------------------------------------------------------------------
template <typename T>
void IPTools<T>::RotateImage_cpu(T *pData, int iW, int iH, T *pOutData, int iOutWidth, int iOutHeight, double theta, int interpolation = 0)
{
	memset(pOutData, 0, sizeof(T) * iOutWidth*iOutHeight);
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

			if (tu >= 0 && tu < iW && tv >= 0 && tv < iH)
			{
				if (interpolation == 0)
					pOutData[row*iOutWidth + col] = pData[(int)tv*iW + (int)tu];
				else
					pOutData[row*iOutWidth + col] = getInterpolatedPixel_TF(tu, tv, iW, iH, pData);
			}
		}
	}

}