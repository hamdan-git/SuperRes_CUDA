#pragma once
#include "Common.h"
#include<time.h>
#include <stdio.h>
#include <Windows.h>

//--------------------------
class XcRect
{
public:
	XcRect() { left = right = top = bottom = z1 = z2 = iWidth = iHeight = 0; }
	XcRect(int iLeft, int iTop, int iRight, int iBottom) { left = iLeft, top = iTop, right = iRight, bottom = iBottom, z1 = 0, z2 = 0; iWidth = iRight - iLeft + 1; iHeight = iBottom - iTop + 1; }
	XcRect(int iLeft, int iTop, int iRight, int iBottom, int iZ1, int iZ2) { left = iLeft, top = iTop, right = iRight, bottom = iBottom, z1 = iZ1, z2 = iZ2; iWidth = iRight - iLeft + 1; iHeight = iBottom - iTop + 1; }

	void sLeft(int iL) { left = iL; iWidth = right - left + 1; }
	void sRight(int iR) { right = iR; iWidth = right - left + 1; }
	void sTop(int iT) { top = iT; iHeight = bottom - top + 1; }
	void sBottom(int iB) { bottom = iB; iHeight = bottom - top + 1; }

	void sWidth(int iW) { iWidth = iW; right = iWidth + left - 1; }
	void sHeight(int iH) { iHeight = iH; bottom = iHeight + top - 1; }

	int gLeft() { return left; }
	int gRight() { return right; }
	int gTop() { return top; }
	int gBottom() { return bottom; }
	int gWidth() { return iWidth; }
	int gHeight() { return iHeight; }
public:
	int left, right, top, bottom, z1, z2;
	int iWidth;
	int iHeight;
};

struct XcIP_Messages
{
	std::string strError;
	std::string strLog;
};

template<typename T>
int SimilarityWindowDefectCorrect_MCP(T *pOrigData, const unsigned char *pOrigMapData, int iWidth, int iHeight, const XcRect &rectLimit, bool bOptimized, int iPooNum)
{
	double *pCurValues = 0;
	int *pIndex = 0;
	try
	{

		int iLargeWin = 6; //10Aug11, changed from 15 to 20
		int iWin = 2;
		if (bOptimized)
		{
			iLargeWin = 4;
			iWin = 2;
		}

		int iWinSize = (2 * iWin + 1) * (2 * iWin + 1);

		unsigned short iLocalLabel = 121; //some label used for flaging indicating it was 11 and already processed

		pCurValues = new double[iWinSize];
		pIndex = new int[iWinSize];
		int iIndex = 0;
		for (int j = -iWin; j <= iWin; j++)
		{
			for (int i = -iWin; i <= iWin; i++)
			{
				pIndex[iIndex] = j * iWidth + i;
				iIndex++;
			}
		}



		//we want to remove the label after a row is processed. basically we want to make the progressive correction by after a row is done 
		std::vector<int> vCorrectedPixelIndex; //x coordinate of the row

		const unsigned char *pMapData = pOrigMapData;
		//T *pInData = InImage.m_ppData[0];

		T *pOutData = pOrigData;
		int iTargetX = -1, iTargetY = -1;
		int iCurLabel = 0;
		//double fValidSum = 0.0;
		float fVal = 0.0;
		float fVal2 = 0.0;
		int iVal = 0;
		//long iOffset = 0;
		for (int iY = rectLimit.top; iY < iHeight && iY <= rectLimit.bottom; iY++)
		{
			long iOffset = iY * iWidth + rectLimit.left;
			pOutData = &pOrigData[iOffset];
			pMapData = &pOrigMapData[iOffset];
			//vCorrectedPixelIndex.clear(); //start again for the next row
			for (int iX = rectLimit.left; iX < iWidth && iX <= rectLimit.right; iX++)
			{

				iCurLabel = *pMapData;
				if (iCurLabel > 0 && iCurLabel != iLocalLabel)
				{
					//prepare the pCurValues
					for (int k = 0; k < iWinSize; k++) pCurValues[k] = -1.0;
					int iCurValidNeighbours = 0;
					iIndex = 0;
					double fValidSum = 0.0;
					for (int j = -iWin; j <= iWin; j++)
					{
						for (int i = -iWin; i <= iWin; i++)
						{
							int iNewX = i + iX;
							int iNewY = j + iY;
							if (iNewX >= 0 && iNewX < iWidth && iNewY >= 0 && iNewY < iHeight)
							{
								unsigned char iLabel = *(pMapData + pIndex[iIndex]);
								if (iLabel == 0 || iLabel == iLocalLabel)
								{
									pCurValues[iIndex] = (double)*(pOutData + pIndex[iIndex]);
									iCurValidNeighbours++;
									fValidSum += pCurValues[iIndex];
								}
							}
							iIndex++;
						}
					}
					double fCurentWinMean = 1.0;
					if (iCurValidNeighbours > 0)
						fCurentWinMean = fValidSum / (double)iCurValidNeighbours;
					double fLowLimitForOptimizedCase = fCurentWinMean * 0.025;
					//double fLowLimitForOptimizedCase = fCurentWinMean * 0.0525;


					int iHalfCurValidNeighbours = (iCurValidNeighbours >> 1) + 1; //added on 10Aug11
					//now go through all around the and check the windows
					int iStartX = iX - iLargeWin + iWin;
					if (iStartX < iWin) iStartX = iWin;
					int iStartY = iY - iLargeWin + iWin;
					if (iStartY < iWin) iStartY = iWin;
					int iEndX = iLargeWin + iX - iWin;
					if (iEndX >= iWidth - iWin) iEndX = iWidth - iWin - 1;
					int iEndY = iLargeWin + iY - iWin;
					if (iEndY >= iHeight - iWin) iEndY = iHeight - iWin - 1;

					double fMinDiff = FLT_MAX;

					T *pLocOutData = &pOrigData[iStartY*iWidth + iStartX];
					const unsigned char *pLocMapData = &pOrigMapData[iStartY*iWidth + iStartX];
					int iJumpX = iWidth - (iEndX - iStartX + 1);
					for (int jj = iStartY; jj <= iEndY; jj++)
					{
						for (int ii = iStartX; ii <= iEndX; ii++)
						{
							if ((ii != iX || jj != iY) && (*pLocMapData == 0 || (*pLocMapData == iLocalLabel && abs(ii - iX) + abs(jj - iY) > iLargeWin - 1))) //ha added on 25Feb2015, cluster size larger than the large win size were not update w/out the last term
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
										if (iNewX >= 0 && iNewX < iWidth && iNewY >= 0 && iNewY < iHeight && pCurValues[iIndex] >= 0.0)
										{
											int iLabel = *(pLocMapData + pIndex[iIndex]);
											if (iLabel == 0 || iLabel == iLocalLabel)
											{
												iLocalValidNeighbours++;
												double fDiff = fabs((double)*(pLocOutData + pIndex[iIndex]) - pCurValues[iIndex]);
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

										///////////////
										//if the latest diff is less than 2.5% of the current window's mean then stop searching
										//In the future, the search for similarity window will be done from nearest neigbours to outward
										if (bOptimized)
										{
											if (fMinDiff < fLowLimitForOptimizedCase)
											{
												jj = iEndY + 1; break;
											}
										}
										//////////////
									}
								}


							}
							pLocMapData++;
							pLocOutData++;

						}
						pLocMapData += iJumpX;
						pLocOutData += iJumpX;
					}
					if (iTargetX >= 0 && iTargetY >= 0)
					{
						fVal = (float)(pOrigData[iTargetY*iWidth + iTargetX]);

						//if ( fVal > 0.0 ) 
						//	fVal2 = sqrt(fVal);
						//else
						//	fVal2 = 0.0;


						*pOutData = (T)(fVal);// + 0.2f * (fVal2 + (float)(5-rand()%10))) ;//  + (iMeanGradient-rand()%iDoubleMeanGradient);
						////kha

					}
					else
					{
						fVal = (float)*pOutData;
						*pOutData = (T)(fVal + 0.2f * (float)(sqrt(fVal) + (5 - rand() % 10)));
					}

				}


				pMapData++;
				//pInData++;
				pOutData++;
			}

		}


	}
	catch (...)
	{
		if (pCurValues) { delete[] pCurValues; pCurValues = 0; }
		if (pIndex) { delete[] pIndex; pIndex = 0; }
		throw;
	}
	if (pCurValues) { delete[] pCurValues; pCurValues = 0; }
	if (pIndex) { delete[] pIndex; pIndex = 0; }
	return 0;
}


//-----------------------------------------------------------------------------------------------------------------
//template<typename T>
class FillGapWinThreadClass
{
public:
	const void *m_pInData;
	void *m_pOutData;
	const unsigned char *m_pMapData;
	int m_iWidth;
	int m_iHeight;
	XcRect m_rectLimit;
	int m_iCurrentPoolNum;
	bool m_bOptimize;
	//XcDataType m_iPixelType;
public:

	FillGapWinThreadClass(const void *pInData, void *pOutData, const unsigned char *pMapData, int iWidth, int iHeight, const XcRect &rectLimit, int iTileColumnIndex, bool bOptimize, int iCurrentPoolNum)
	{
		m_pInData = pInData;
		m_pOutData = pOutData;
		m_pMapData = pMapData;
		m_rectLimit = rectLimit;
		m_iWidth = iWidth;
		m_iHeight = iHeight;
		m_iCurrentPoolNum = iCurrentPoolNum;
		m_bOptimize = bOptimize;

	}
};


//template<typename T>
DWORD WINAPI WinMTS_FillGapForGeometricCorrection_Proc_Float(LPVOID pParam)
{
	FillGapWinThreadClass *pObject = (FillGapWinThreadClass *)pParam;
	if (pObject == NULL /*|| !pObject->IsKindOf(RUNTIME_CLASS(CMyObject))*/)
		return 0;
//	switch (pObject->m_iPixelType)
//	{
 SimilarityWindowDefectCorrect_MCP<float>((float*)pObject->m_pOutData, pObject->m_pMapData, pObject->m_iWidth, pObject->m_iHeight, pObject->m_rectLimit, pObject->m_bOptimize, pObject->m_iCurrentPoolNum); 
//	};


	return 1;
}

//UNTIL HERE
//---------------------------------------------------------------------------------------------------------------------------

short FillGapForGeoCor_MultiThreading(const float* pInData, float *pOutData, unsigned char *pMapData, int iWidth, int iHeight, const std::vector<int> &vEdges, bool bOptimize, int iPoolSize)
{
	int iResult = 1;

	XcRect rectLimit(0, 0, iWidth - 1, iHeight - 1);
	HANDLE *hWinThread = nullptr;
	std::vector<FillGapWinThreadClass *> vpThreadClasses;
	//laa
	try
	{
		memcpy(pOutData, pInData, sizeof(float)*iWidth*iHeight);
		if (iPoolSize <= 0) //no multi core
		{
			SimilarityWindowDefectCorrect_MCP<float>((float*)pOutData, pMapData, iWidth, iHeight, rectLimit, bOptimize, 0);
			

		}
		else
		{
			int iThreadPriority = HIGH_PRIORITY_CLASS & THREAD_PRIORITY_HIGHEST;


			//set the average image to have the same background as input
			long iFrameSize = iWidth * iHeight;

			//get the pool size according to the number of edges 
			iPoolSize = (int)vEdges.size();//iWidth / iTileWidth - 1;

			//printf("poolsize for MS simWin %d\n", iPoolSize);

			hWinThread = new HANDLE[iPoolSize];
			//int iEdgeIndex = iTileWidth;
			bool bWorkerThreads = true;
			for (int i = 0; i < iPoolSize; i++)
			{

				rectLimit.left = max(vEdges.at(i) - 2, 0);
				rectLimit.right = min(vEdges.at(i) + 2, iWidth - 1);
				//rectLimit.left = max(iEdgeIndex-iTileWidth, 0);
				//rectLimit.right = min(iEdgeIndex, iWidth-1);
				FillGapWinThreadClass *myStruc = new FillGapWinThreadClass((void*)pInData, (void*)pOutData, pMapData, iWidth, iHeight, rectLimit, vEdges.at(i), bOptimize, i);
				vpThreadClasses.push_back(myStruc); //we need to delete it later

				//need to have a 
				HANDLE hThread = CreateThread(NULL, 0, WinMTS_FillGapForGeometricCorrection_Proc_Float, myStruc, 0, NULL);//pThread->m_hThread;
				if (hThread == NULL)
				{
					bWorkerThreads = false;
					break;
				}
				SetThreadPriority(hThread, iThreadPriority);
				ResumeThread(hThread);
				hWinThread[i] = hThread;

				//iEdgeIndex += iTileWidth;
			}


			DWORD iWaitLimitTime = 50000; //set limit to 5 second, it can be INFINITE to wait for infinte time
			DWORD iR = WaitForMultipleObjects(iPoolSize, hWinThread, TRUE, iWaitLimitTime);
			if (iR != WAIT_OBJECT_0 || !bWorkerThreads) //at least one thread did not finish, so do the normal averaging
			{
				for (int i = 0; i < iPoolSize; i++)
				{
					CloseHandle(hWinThread[i]);
				}
				delete[] hWinThread; hWinThread = 0;

				printf("Failed thread in MC sim win\n");
				rectLimit = XcRect(0, 0, iWidth - 1, iHeight - 1);
				SimilarityWindowDefectCorrect_MCP<float>((float*)pOutData, pMapData, iWidth, iHeight, rectLimit, bOptimize, 0); 
				

			}

		}

	}
	catch (short a)
	{
		if (hWinThread)
		{
			//need to close the thread handles
			for (int i = 0; i < iPoolSize; i++)
				CloseHandle(hWinThread[i]);
			delete[] hWinThread;
		}

		for (int k = 0; k < (int)vpThreadClasses.size(); k++) {
			if (vpThreadClasses[k] != nullptr) { delete vpThreadClasses[k]; vpThreadClasses[k] = 0; }
		}
		vpThreadClasses.clear();

		throw(a);
	}

	if (hWinThread)
	{
		//need to close the thread handles
		for (int i = 0; i < iPoolSize; i++)
			CloseHandle(hWinThread[i]);
		delete[] hWinThread; hWinThread = 0;
	}
	for (int k = 0; k < (int)vpThreadClasses.size(); k++) {
		if (vpThreadClasses[k] != nullptr) { delete vpThreadClasses[k]; vpThreadClasses[k] = 0; }
	}
	vpThreadClasses.clear();

	return iResult;
}


//------------------------------------------
template<typename T>
int SimilarityWindowDefectCorrect_MCP2(T *pOrigData, const unsigned char *pOrigMapData, int iWidth, int iHeight, bool bOptimized, int iPooNum)
{
	double *pCurValues = 0;
	int *pIndex = 0;
	try
	{

		int iLargeWin = 6; //10Aug11, changed from 15 to 20
		int iWin = 2;
		//if (bOptimized)
		//{
		//	iLargeWin = 4;
		//	iWin = 2;
		//}

		int iWinSize = (2 * iWin + 1) * (2 * iWin + 1);

		//unsigned short iLocalLabel = 121; //some label used for flaging indicating it was 11 and already processed

		pCurValues = new double[iWinSize];
		pIndex = new int[iWinSize];
		int iIndex = 0;
		for (int j = -iWin; j <= iWin; j++)
		{
			for (int i = -iWin; i <= iWin; i++)
			{
				pIndex[iIndex] = j * iWidth + i;
				iIndex++;
			}
		}



		//we want to remove the label after a row is processed. basically we want to make the progressive correction by after a row is done 
		std::vector<int> vCorrectedPixelIndex; //x coordinate of the row

		const unsigned char *pMapData = pOrigMapData;
		//T *pInData = InImage.m_ppData[0];

		T *pOutData = pOrigData;
		int iTargetX = -1, iTargetY = -1;
		int iCurLabel = 0;
		//double fValidSum = 0.0;
		float fVal = 0.0;
		float fVal2 = 0.0;
		int iVal = 0;
		//long iOffset = 0;
		for (int iY = 0; iY < iHeight ; iY++)
		{
			long iOffset = iY * iWidth ;
			pOutData = &pOrigData[iOffset];
			pMapData = &pOrigMapData[iOffset];
			//vCorrectedPixelIndex.clear(); //start again for the next row
			for (int iX = 0; iX < iWidth ; iX++)
			{

				iCurLabel = *pMapData;
				if (iCurLabel > 0 /*&& iCurLabel != iLocalLabel*/)
				{
					//prepare the pCurValues
					for (int k = 0; k < iWinSize; k++) pCurValues[k] = -1.0;
					int iCurValidNeighbours = 0;
					iIndex = 0;
					double fValidSum = 0.0;
					for (int j = -iWin; j <= iWin; j++)
					{
						for (int i = -iWin; i <= iWin; i++)
						{
							int iNewX = i + iX;
							int iNewY = j + iY;
							if (iNewX >= 0 && iNewX < iWidth && iNewY >= 0 && iNewY < iHeight)
							{
								unsigned char iLabel = *(pMapData + pIndex[iIndex]);
								if (iLabel == 0 /*|| iLabel == iLocalLabel*/)
								{
									pCurValues[iIndex] = (double)*(pOutData + pIndex[iIndex]);
									iCurValidNeighbours++;
									fValidSum += pCurValues[iIndex];
								}
							}
							iIndex++;
						}
					}
					double fCurentWinMean = 1.0;
					if (iCurValidNeighbours > 0)
						fCurentWinMean = fValidSum / (double)iCurValidNeighbours;
					double fLowLimitForOptimizedCase = fCurentWinMean * 0.025;
					//double fLowLimitForOptimizedCase = fCurentWinMean * 0.0525;


					int iHalfCurValidNeighbours = (iCurValidNeighbours >> 1) + 1; //added on 10Aug11
					//now go through all around the and check the windows
					int iStartX = iX - iLargeWin + iWin;
					if (iStartX < iWin) iStartX = iWin;
					int iStartY = iY - iLargeWin + iWin;
					if (iStartY < iWin) iStartY = iWin;
					int iEndX = iLargeWin + iX - iWin;
					if (iEndX >= iWidth - iWin) iEndX = iWidth - iWin - 1;
					int iEndY = iLargeWin + iY - iWin;
					if (iEndY >= iHeight - iWin) iEndY = iHeight - iWin - 1;

					double fMinDiff = FLT_MAX;

					T *pLocOutData = &pOrigData[iStartY*iWidth + iStartX];
					const unsigned char *pLocMapData = &pOrigMapData[iStartY*iWidth + iStartX];
					int iJumpX = iWidth - (iEndX - iStartX + 1);
					for (int jj = iStartY; jj <= iEndY; jj++)
					{
						for (int ii = iStartX; ii <= iEndX; ii++)
						{
							if ((ii != iX || jj != iY) && (*pLocMapData == 0 /*|| (*pLocMapData == iLocalLabel && abs(ii - iX) + abs(jj - iY) > iLargeWin - 1)*/)) //ha added on 25Feb2015, cluster size larger than the large win size were not update w/out the last term
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
										if (iNewX >= 0 && iNewX < iWidth && iNewY >= 0 && iNewY < iHeight && pCurValues[iIndex] >= 0.0)
										{
											int iLabel = *(pLocMapData + pIndex[iIndex]);
											if (iLabel == 0 /*|| iLabel == iLocalLabel*/)
											{
												iLocalValidNeighbours++;
												double fDiff = fabs((double)*(pLocOutData + pIndex[iIndex]) - pCurValues[iIndex]);
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
							pLocMapData++;
							pLocOutData++;

						}
						pLocMapData += iJumpX;
						pLocOutData += iJumpX;
					}
					if (iTargetX >= 0 && iTargetY >= 0)
					{
						fVal = (float)(pOrigData[iTargetY*iWidth + iTargetX]);

						*pOutData = (T)(fVal);// + 0.2f * (fVal2 + (float)(5-rand()%10))) ;//  + (iMeanGradient-rand()%iDoubleMeanGradient);
						////kha

					}
					//else
					//{
					//	fVal = (float)*pOutData;
					//	*pOutData = (T)(fVal + 0.2f * (float)(sqrt(fVal) + (5 - rand() % 10)));
					//}

				}


				pMapData++;
				//pInData++;
				pOutData++;
			}

		}


	}
	catch (...)
	{
		if (pCurValues) { delete[] pCurValues; pCurValues = 0; }
		if (pIndex) { delete[] pIndex; pIndex = 0; }
		throw;
	}
	if (pCurValues) { delete[] pCurValues; pCurValues = 0; }
	if (pIndex) { delete[] pIndex; pIndex = 0; }
	return 0;
}

//--------------------------------------------------------------
void TestSimilarityWindows_cpu(float *pData, unsigned char *pMapData, int iW, int iH, int iD)
{
	int iFrameSize = iW * iH;
	float *pOutData = new float[iFrameSize*iD];

	

	clock_t start = clock();
	for (int i = 0; i < iD; i++)
	{
		memcpy(&pOutData[i*iFrameSize], &pData[i*iFrameSize], sizeof(float)*iFrameSize);
		SimilarityWindowDefectCorrect_MCP2<float>(&pOutData[i*iFrameSize], pMapData, iW, iH, 0, 1);
	}

	printf("SimWin cpu took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("C:\\Temp\\SimWin_cpu.raw", pOutData, iW, iH, iD);

	delete[] pOutData;
}

//--------------------------------------------------------------
void TestSimilarityWindows_MC_cpu(float *pData, unsigned char *pMapData, int iW, int iH, int iD, const std::vector<int> &vEdges)
{
	int iFrameSize = iW * iH;
	float *pOutData = new float[iFrameSize*iD];



	clock_t start = clock();
	for (int i = 0; i < iD; i++)
	{

		FillGapForGeoCor_MultiThreading((const float*)&pData[i*iFrameSize], &pOutData[i*iFrameSize], pMapData, iW, iH, vEdges, false, 1);
	}

	printf("SimWin cpu MC took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("C:\\Temp\\SimWin_MC cpu.raw", pOutData, iW, iH, iD);

	delete[] pOutData;
}

//-----------------------------------------------
void TestSimilarityWindows_gpu(float *pData, unsigned char *pMapData, int iW, int iH)
{
	int iFrameSize = iW * iH;
	float *pOutData = new float[iFrameSize];

	memcpy(pOutData, pData, sizeof(float)*iFrameSize);

	addWithCuda();


	clock_t start = clock();
	for (int i = 0; i < iIterations; i++)
	{
		SimWin_Cuda(pData, pOutData, iW, iH, pMapData);
	}

	printf("SimWin cpu took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("C:\\Temp\\SimWin_gpu.raw", pOutData, iW, iH);

	delete[] pOutData;
}

//-----------------------------------------------
void TestSimilarityWindows_tex_gpu(float *pData, unsigned char *pMapData, int iW, int iH, int iD)
{
	int iFrameSize = iW * iH;
	float *pOutData = new float[iFrameSize*iD];

	

	addWithCuda();


	clock_t start = clock();
	
	//for (int i = 0; i < iD; i++)
	//{

	//	SimWin_tex_Cuda(&pData[i*iFrameSize], &pOutData[i*iFrameSize], iW, iH, pMapData);
	//}

	SimWin_tex_Cuda(pData, pOutData, iW, iH, iD, pMapData);

	printf("SimWin cpu took average of %f s\n", (double)(clock() - start) / (double)CLOCKS_PER_SEC);

	WriteRawData("C:\\Temp\\SimWin_tex_gpu.raw", pOutData, iW, iH, iD);

	delete[] pOutData;
}



//--------------------------
void TestSimilarityWindowsCorrection()
{
	int iWidth = 2048;// 1024;
	int iHeight = 2048;// 256;
	int iNumFrames = 100;
	int iFrameSize = iWidth * iHeight;
	float *pInImage = ReadRawData<float>("C:\\Temp\\TestSimWin.raw", iWidth, iHeight);
	float *pInImage2 = new float[iFrameSize * iNumFrames];

	for (int z = 0; z < iNumFrames; z++)
	{
		memcpy(&pInImage2[z*iFrameSize], pInImage, sizeof(float)*iFrameSize);
	}

	unsigned char *pMapData = new unsigned char[iFrameSize];
	memset(pMapData, 0, iFrameSize);
	std::vector<int> vEdge;
	for (int iY = 0; iY < iHeight; iY++)
	{

		pMapData[iY*iWidth + 760] = 1; 
		pMapData[iY*iWidth + 660] = 1; 
		pMapData[iY*iWidth + 560] = 1; 
		pMapData[iY*iWidth + 460] = 1; 
		pMapData[iY*iWidth + 360] = 1; 
		pMapData[iY*iWidth + 260] = 1; 
		pMapData[iY*iWidth + 291] = 1; 
		pMapData[iY*iWidth + 325] = 1;
		pMapData[iY*iWidth + 160] = 1;
		//for (int i = 20; i < 150; i+=10)
		//	pMapData[iY*iWidth + i] = 1;
	}

	vEdge.push_back(760);
	 vEdge.push_back(660);
	 vEdge.push_back(560);
	 vEdge.push_back(460);
	 vEdge.push_back(360);
	 vEdge.push_back(260);
	 vEdge.push_back(291);
	 vEdge.push_back(325);
	 vEdge.push_back(160);
	 //for (int i = 20; i < 150; i += 10)
		// vEdge.push_back(i);

	if (pInImage == NULL) { printf("null input"); exit(0); }

	TestSimilarityWindows_cpu(pInImage2, pMapData, iWidth, iHeight, iNumFrames);
	//TestSimilarityWindows_gpu(pInImage, pMapData, iWidth, iHeight);
	TestSimilarityWindows_tex_gpu(pInImage2, pMapData, iWidth, iHeight, iNumFrames);
	TestSimilarityWindows_MC_cpu(pInImage2, pMapData, iWidth, iHeight, iNumFrames, vEdge);

	delete[] pInImage;
	delete[] pInImage2;
	delete[] pMapData;

}