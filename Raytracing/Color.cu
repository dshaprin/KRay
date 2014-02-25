#include "color.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

unsigned char SRGB_COMPRESS_CACHE[4097];
__constant__ unsigned char  SRGB_COMPRESS_CACHE_DEV[4097];
void initColor(void)
{
	// precache the results of convertTo8bit_sRGB, in order to avoid the costly pow()
	// in it and use a lookup table instead, see Color::convertTo8bit_sRGB_cached().
	for (int i = 0; i <= 4096; i++)
		SRGB_COMPRESS_CACHE[i] = (unsigned char) convertTo8bit_sRGB(i / 4096.0f);
	HANDLE_ERROR(cudaMemcpyToSymbol(SRGB_COMPRESS_CACHE_DEV, SRGB_COMPRESS_CACHE, sizeof(unsigned char) * 4097));
}

unsigned __device__  convertTo8bit_sRGB_cached(float x)
{
	if (x <= 0) return 0;
	if (x >= 1) return 255;
	return SRGB_COMPRESS_CACHE_DEV[int(x * 4096.0f)];
}
inline unsigned __device__ __host__  convertTo8bit(float x)
{
	if (x < 0) x = 0;
	if (x > 1) x = 1;
	return nearestInt(x * 255.0f);
}