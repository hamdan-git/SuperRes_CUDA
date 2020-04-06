#pragma once
#include <stdio.h>

const int iIterations = 10;

//---------------------------------------------------------------------
template<typename T>
T *ReadRawData(char *filename, int iW, int iH)
{
	FILE *in;
	fopen_s(&in, filename, "rb");
	if (in == NULL)
	{
		printf("Colud not read input image %s\n", filename);
		return NULL;
	}
	int iFrameSize = iW * iH;
	T *pData = new T[iFrameSize];
	
	
		if (fread(pData, sizeof(T), iFrameSize, in) != iFrameSize)
		{
			printf("Colud not read input image %s due to size mismatch\n", filename);
			delete[] pData;
			return NULL;
		}
	
	fclose(in);
	return pData;
}

//---------------------------------------------------------------------
template<typename T>
T *ReadRawData(char *filename, int iW, int iH, int iD)
{
	FILE *in;
	fopen_s(&in, filename, "rb");
	if (in == NULL)
	{
		printf("Colud not read input image %s\n", filename);
		return NULL;
	}
	int iFrameSize = iW * iH;
	T *pData = new T[iFrameSize*iD];

	for (int z = 0; z < iD; z++)
	{
		if (fread(&pData[iFrameSize*z], sizeof(T), iFrameSize, in) != iFrameSize)
		{
			printf("Colud not read input image %s due to size mismatch\n", filename);
			delete[] pData;
			fclose(in);
			return NULL;
		}
	}
	fclose(in);
	return pData;
}
//---------------------------------------------------------------------
template<typename T>
bool WriteRawData(char *filename, T *pData, int iW, int iH)
{
	FILE *out;
	fopen_s(&out, filename, "wb");
	if (out == NULL) return NULL;
	int iFrameSize = iW * iH;

	if (fwrite(pData, sizeof(T), iFrameSize, out) != iFrameSize)
	{
		fclose(out);
		return false;
	}
	fclose(out);
	return true;
}

//---------------------------------------------------------------------
template<typename T>
bool WriteRawData(char *filename, T *pData, int iW, int iH, int iD)
{
	FILE *out;
	fopen_s(&out, filename, "wb");
	if (out == NULL) return NULL;
	int iFrameSize = iW * iH;

	for (int iZ = 0; iZ < iD; iZ++)
	{
		if (fwrite(&pData[iZ*iFrameSize], sizeof(T), iFrameSize, out) != iFrameSize)
		{
			fclose(out);
			return false;
		}
	}
	fclose(out);
	return true;
}



