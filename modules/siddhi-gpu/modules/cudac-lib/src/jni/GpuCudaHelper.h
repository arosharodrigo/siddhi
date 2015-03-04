/*
 * GpuCudaHelper.h
 *
 *  Created on: Jan 23, 2015
 *      Author: prabodha
 */

#ifndef GPUCUDAHELPER_H_
#define GPUCUDAHELPER_H_

#include <stdlib.h>
#include <stdio.h>

namespace SiddhiGpu
{

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error [%s] at line [%d] in file [%s]\n",			\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		cudaDeviceReset();                                                  \
		exit(1);															\
	} }

#define CUDA_CHECK_WARN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error [%s] at line [%d] in file [%s]\n",			\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
	} }

#define CUDA_CHECK_SET_FLAG(value, flag) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error [%s] at line [%d] in file [%s]\n",			\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		flag = true;                                                        \
	} else {                                                                \
        flag = false;                                                       \
	}}

class GpuCudaHelper
{
public:
	static bool SelectDevice(int _iDeviceId, const char * _zTag, FILE * _fpLog);
	static void GetDeviceId(const char * _zTag, FILE * _fpLog);
	static void DeviceReset();

	static void AllocateHostMemory(bool _bPinGenericMemory, char ** _ppAlloc, char ** _ppAlignedAlloc, unsigned int _iAllocSize, const char * _zTag, FILE * _fpLog);
	static void FreeHostMemory(bool _bPinGenericMemory, char ** _ppAlloc, char ** _ppAlignedAlloc, unsigned int _iAllocSize, const char * _zTag, FILE * _fpLog);

};

}


#endif /* GPUCUDAHELPER_H_ */
