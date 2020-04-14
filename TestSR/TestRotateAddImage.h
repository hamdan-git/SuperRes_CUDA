#pragma once
#include "Common.h"
#include<time.h>
#include <stdio.h>
#include <math.h>
#include <ipp.h>
#include <ippi.h>
#include <ipps.h>
#include <algorithm>
#include <string>
#include <iostream>
#include "IPTools.h"
#include "CommonImage.h"

//---------------------------------------------------------------------
void PrintHelp()
{
	std::cout << "Hint\n";
	std::cout << "exe <tif_input_filename> <tif_output_filename> options\n";
	std::cout << "-a angle_val\ttilt angle\n";
	std::cout << "-s step_val\tstep\n";
	std::cout << "-f or -r\tforward or backward regarding the input image acquition\n";
	std::cout << "-t <algo type>\tgpu or gpu_lut using look up table\n";
	std::cout << "-m mag_val\tmagnification\n";
}

//----------------------------------------------------------------------------
void FindDimentionAfterRotationAdd(int iInWidth, int iInHeight, double theta, double fMagnification, int &iOutWidth, int &iOutHeight)
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

	iOutWidth *= fMagnification;
	iOutHeight *= fMagnification;
}


//--------------------------------------------------------------
unsigned char * TestRotateAdd_gpu(unsigned char *pData, int iW, int iH, int iD, int pixType, int &iOutWidth, int &iOutHeight, double theta, float fShiftStep, double fMag, bool bReversed, bool bUseLUT)
{
	unsigned char *pOutData = NULL;
	unsigned short *pOutDataKir = NULL;
	int iRotWidth = iW;
	int iRotHeight = iH;

	FindDimentionAfterRotationAdd(iW, iH, theta, fMag, iRotWidth, iRotHeight);

	iOutWidth = iRotWidth;
	iOutHeight = iRotHeight + iD * fShiftStep + 2;;

	int pixSize = 2;
	switch (pixType)
	{
	case CIMG_PIXEL_TYPE_USG_8: pixSize = 1; break;
	case CIMG_PIXEL_TYPE_USG_16: pixSize = 2; break;
	case CIMG_PIXEL_TYPE_USG_32: pixSize = 4; break;
	case CIMG_PIXEL_TYPE_FLOAT: pixSize = 4; break;
	};

	int iOutFrameSize = iOutWidth * iOutHeight;

	pOutData = new unsigned char[iOutFrameSize*pixSize];
	pOutDataKir = new unsigned short[iOutFrameSize];

	memset(pOutData, 0, pixSize*iOutFrameSize);

	addWithCuda();


	clock_t start = clock();
	if (bUseLUT)
		RotateAddImage_lut_Cuda(pData, iW, iH, iD, pOutData, iOutWidth, iOutHeight, pixType, theta, fShiftStep, fMag, bReversed);
	else
		RotateAddImage_Cuda(pData, iW, iH, iD, pOutData, iOutWidth, iOutHeight, pixType, theta, fShiftStep, fMag, bReversed);


	printf("Rotate gpu (tex) took average of %f s  (%d, %d)\n", ((double)(clock() - start) / (double)CLOCKS_PER_SEC) , iOutWidth, iOutHeight);

	//WriteRawData<unsigned char>("c:\\Temp\\RotateAddImage_gpu.raw", pOutData, iOutWidth*pixSize, iOutHeight);
	//WriteRawData<unsigned short>("c:\\Temp\\RotateAddImage_gpu.raw", pOutDataKir, iOutWidth, iOutHeight);
	//WriteRawData<unsigned char>("c:\\Temp\\RotateAddImage_gpu.raw", pData, iW*pixSize, iH, iD);


	//delete[] pOutData;
	//delete[] pMaskData;
	return pOutData;
}

//cudaError_t Filtering_1D_Cuda(float *pInData, float *pOutData, int iWidth, int iHeight, float *pKernel, int iKernelSize, int dir)

//--------------------------
void TestRotateAddImage(int argc,  char *argv[])
{
#ifdef _DEBUG
	int iWidth = 1024;// 2048;// 1024;
	int iHeight = 256;// 64;// 256;
	int iNumFrames = 1882;// 2000;
	double fTheta = 5.45;// -15.5;
	double fShiftScale = 0.45;// 0.65;
	double fMagnification = 1.0;

	fShiftScale *= fMagnification;

	//unsigned short *pInImage = ReadRawData<unsigned short>("D:\\Images\\XCounter\\Customers\\0_Internal\\Hamdan\\SuperRes\\Detector2\\Target\\frame_200hz_ang_-15.5_angled_target_gainCor_u16.raw", iWidth, iHeight, iNumFrames);
	unsigned short *pInImage = ReadRawData<unsigned short>("D:\\Images\\XCounter\\Customers\\0_Internal\\Mattias\\SuperRes\\Speed_5_110Hz_1024X256X1802_u16.raw", iWidth, iHeight, iNumFrames);
	if (pInImage == NULL) { printf("null input"); exit(0); }


	//TestRotate_cpu(pInImage, iWidth, iHeight, fTheta);
	TestRotateAdd_gpu(pInImage, iWidth, iHeight, iNumFrames, fTheta, fShiftScale, fMagnification);
	delete[] pInImage;
#else
	if (argc < 3)
	{
		PrintHelp();
		exit(0);
	}
	std::string strFilename(argv[1]);
	std::string strFilenameOut(argv[2]);
	std::string strAlgoType = "gpu_lut"; //default
	double fTheta = 0.0;
	double fStep = 1.0;
	double fMagnification = 1.0;
	bool bReversed = false;
	for (int i = 2; i < argc; i++)
	{
		if (strcmp(argv[i], "-a") == 0)
		{
			fTheta = atof(argv[i + 1]);
			i++;
		}
		else if (strcmp(argv[i], "-s") == 0)
		{
			fStep = atof(argv[i + 1]);
			i++;
		}
		else if (strcmp(argv[i], "-m") == 0)
		{
			fMagnification = atof(argv[i + 1]);
			i++;
		}
		else if (strcmp(argv[i], "-f") == 0)
		{
			bReversed = false;
		}
		else if (strcmp(argv[i], "-r") == 0)
		{
			bReversed = true;
		}
		else if (strcmp(argv[i], "-t") == 0)
		{
			strAlgoType = std::string(argv[i + 1]);
			i++;
		}
	}

	std::cout << "To do!\n";
	std::cout << "Input file = " << strFilename << "\n";
	std::cout << "Theta = " << fTheta << "\n";
	std::cout << "Steps = " << fStep << "\n";
	std::cout << "Magnification = " << fMagnification << "\n";
	bReversed == true ? std::cout << "revered scanning\n" : std::cout << "forward scanning\n";
	std::cout << "Type = " << strAlgoType << "\n";

	fStep *= fMagnification;

	unsigned char *pOutData = NULL;
	try
	{
		bool bUseLUT = true;
		if (strAlgoType.compare("gpu") == 0)
			bUseLUT = false;

		CIMG_RC cm_rc;
		CIMG cimg;
		if ((cm_rc = cimg.ReadDataFromImageFile(strFilename.c_str())) != CIMG_RC_OK)
		{
			printf("Problem read readinf tiff file %s with error %d\n", strFilename.c_str(), cm_rc);
			exit(0);
		}
		CIMG_Header header = cimg.header;

		int iOutWidth = 0, iOutHeight = 0;
		pOutData = TestRotateAdd_gpu((unsigned char*)cimg.GetDataPointer(), header.width, header.height, header.num_frames, header.pixType, iOutWidth, iOutHeight,  fTheta, fStep, fMagnification, bReversed, bUseLUT);
		if (pOutData == NULL) throw new std::exception("Returned result image is invalid");
		if (iOutWidth<=0 || iOutHeight<=0) throw new std::exception("Returned result image dimesion is invalid");

		int pixSize = 2;
		switch (header.pixType)
		{
		case CIMG_PIXEL_TYPE_USG_8: pixSize = 1; break;
		case CIMG_PIXEL_TYPE_USG_16: pixSize = 2; break;
		case CIMG_PIXEL_TYPE_USG_32: pixSize = 4; break;
		case CIMG_PIXEL_TYPE_FLOAT: pixSize = 4; break;
		};
		CIMG outImage;
		outImage.Create(iOutWidth, iOutHeight, 1, header.pixType);
		memcpy(outImage.GetDataPointer(), pOutData, pixSize * iOutWidth *iOutHeight);
		outImage.WriteDataToTifFile(strFilenameOut.c_str());

	}
	catch (std::exception ex)
	{
		std::cout << ex.what();
	}
	if (pOutData != NULL) delete[] pOutData;
#endif
}