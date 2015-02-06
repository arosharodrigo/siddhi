/*
 * GpuCudaHelper.cpp
 *
 *  Created on: Jan 23, 2015
 *      Author: prabodha
 */

#include "GpuCudaHelper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>

namespace SiddhiGpu
{

#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

bool GpuCudaHelper::SelectDevice(int _iDeviceId, const char * _zTag, FILE * _fpLog)
{
	fprintf(_fpLog, "[GpuCudaHelper] <%s> Selecting CUDA device\n", _zTag);
	int iDevCount = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&iDevCount));
	fprintf(_fpLog, "[GpuCudaHelper] <%s> CUDA device count : %d\n", _zTag, iDevCount);

	if(iDevCount == 0)
	{
		fprintf(_fpLog, "[GpuCudaHelper] <%s> No CUDA devices found\n", _zTag);
		fflush(_fpLog);
		return false;
	}

	if(_iDeviceId < iDevCount)
	{
//		CUDA_CHECK_WARN(cudaSetDeviceFlags(cudaDeviceMapHost));
		cudaGetLastError();
		CUDA_CHECK_RETURN(cudaSetDevice(_iDeviceId));
		fprintf(_fpLog, "[GpuCudaHelper] <%s> CUDA device set to %d\n", _zTag, _iDeviceId);
		fflush(_fpLog);
		return true;
	}
	fprintf(_fpLog, "[GpuCudaHelper] <%s> CUDA device id %d is wrong\n", _zTag, _iDeviceId);
	fflush(_fpLog);
	return false;
}

void GpuCudaHelper::GetDeviceId(const char * _zTag, FILE * _fpLog)
{
	int iDeviceId = -1;
	CUDA_CHECK_RETURN(cudaGetDevice(&iDeviceId));

	fprintf(_fpLog, "[GpuCudaHelper] <%s> Current CUDA device id %d\n", _zTag, iDeviceId);
	fflush(_fpLog);
}

void GpuCudaHelper::DeviceReset()
{
	CUDA_CHECK_RETURN(cudaDeviceReset());
}

void GpuCudaHelper::AllocateHostMemory(bool _bPinGenericMemory, char ** _ppAlloc, char ** _ppAlignedAlloc, int _iAllocSize, const char * _zTag, FILE * _fpLog)
{
#if CUDART_VERSION >= 4000

    if (_bPinGenericMemory)
    {
        // allocate a generic page-aligned chunk of system memory
        fprintf(_fpLog, "[GpuCudaHelper] <%s> AllocateHostMemory (generic page-aligned system memory) using mmap : %4.2f Mbytes \n",
        		_zTag, (float)_iAllocSize/1048576.0f);
        *_ppAlloc = (char *) mmap(NULL, (_iAllocSize + MEMORY_ALIGNMENT), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);

        *_ppAlignedAlloc = (char *)ALIGN_UP(*_ppAlloc, MEMORY_ALIGNMENT);

        fprintf(_fpLog, "[GpuCudaHelper] <%s> cudaHostRegister registering generic allocated system memory\n", _zTag);
        // pin allocate memory
        CUDA_CHECK_RETURN(cudaHostRegister(*_ppAlignedAlloc, _iAllocSize, cudaHostRegisterMapped));
    }
    else
#endif
    {
        fprintf(_fpLog, "[GpuCudaHelper] <%s> AllocateHostMemory (system memory) using cudaMallocHost : %4.2f Mbytes\n",
        		_zTag, (float)_iAllocSize/1048576.0f);
        // allocate host memory (pinned is required for achieve asynchronicity)
        CUDA_CHECK_RETURN(cudaMallocHost((void **)_ppAlloc, _iAllocSize));
        *_ppAlignedAlloc = *_ppAlloc;
    }
}

void GpuCudaHelper::FreeHostMemory(bool _bPinGenericMemory, char ** _ppAlloc, char ** _ppAlignedAlloc, int _iAllocSize, const char * _zTag, FILE * _fpLog)
{
#if CUDART_VERSION >= 4000

    // CUDA 4.0 support pinning of generic host memory
    if (_bPinGenericMemory)
    {
    	fprintf(_fpLog, "[GpuCudaHelper] <%s> FreeHostMemory (generic page-aligned system memory) using mmap : %4.2f Mbytes \n",
    			_zTag, (float)_iAllocSize/1048576.0f);
        // unpin and delete host memory
    	CUDA_CHECK_RETURN(cudaHostUnregister(*_ppAlignedAlloc));
        munmap(*_ppAlloc, _iAllocSize);
    }
    else
#endif
    {
    	fprintf(_fpLog, "[GpuCudaHelper] <%s> FreeHostMemory (system memory) using cudaMallocHost : %4.2f Mbytes\n",
    			_zTag, (float)_iAllocSize/1048576.0f);
    	CUDA_CHECK_RETURN(cudaFreeHost(*_ppAlloc));
    }

    _ppAlignedAlloc = NULL;
    _ppAlloc = NULL;
}

}


