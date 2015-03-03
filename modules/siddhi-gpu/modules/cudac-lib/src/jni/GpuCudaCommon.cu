#ifndef _GPU_CUDA_COMMON_CU__
#define _GPU_CUDA_COMMON_CU__

#include <stdio.h>

namespace SiddhiGpu
{

__device__ bool cuda_strcmp(const char *s1, const char *s2)
{
//	if(!s1 || !s2) return false; TODO: uncomment

	for ( ; *s1==*s2; ++s1, ++s2) {
		if (*s1=='\0') return true;
	}
	return false;
}

__device__ bool cuda_prefix(char *s1, char *s2)
{
	if(!s1 || !s2) return false;

	for ( ; *s1==*s2; ++s1, ++s2) {
		if (*(s2+1)=='\0') return true;
	}
	return false;
}

__device__ bool cuda_contains(const char *s1, const char *s2)
{
	if(!s1 || !s2) return false;

	int size1 = 0;
	int size2 = 0;

	while (s1[size1]!='\0')
		size1++;

	while (s2[size2]!='\0')
		size2++;

	if (size1==size2)
		return cuda_strcmp(s1, s2);

	if (size1<size2)
		return false;

	for (int i=0; i<size1-size2+1; i++)
	{
		bool failed = false;
		for (int j=0; j<size2; j++)
		{
			if (s1[i+j-1]!=s2[j])
			{
				failed = true;
				break;
			}
		}
		if (! failed)
			return true;
	}
	return false;
}

}

#endif
