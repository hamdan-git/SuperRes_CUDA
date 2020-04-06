#pragma once

//----------------------------------------------------------------------------
void FindDimensionAfterRotation(int iInWidth, int iInHeight, double theta, int &iOutWidth, int &iOutHeight);

template<typename T>
class IPTools
{
public:
	static void DoErosion(const T *pInData, T* pOutData, int iWidth, int iHeight, int iWin);
	static void GetDistanceMap(const T* pInData, int iWidth, int iHeight, T *pOutData = 0, float *pFloatData = 0);
	static void RotateImage_cpu(T *pData, int iW, int iH, T *pOutData, int iOutWidth, int iOutHeight, double theta, int interpolation = 0);
};

template class IPTools<float>;
template class IPTools<double>;
template class IPTools<unsigned short>;
template class IPTools<short>;
template class IPTools<unsigned int>;
template class IPTools<unsigned char>;
template class IPTools<int>;
template class IPTools<unsigned char>;
//--------------------------------------------------------------
