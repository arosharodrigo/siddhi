#ifndef _GPU_CUDA_TIMER_CU__
#define _GPU_CUDA_TIMER_CU__

#include <stdio.h>
#include <stdlib.h>
#include "GpuCudaTimer.h"
#include "GpuCudaHelper.h"

namespace SiddhiGpu
{

GpuCudaTimer::GpuCudaTimer()
{
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&end));
	CUDA_CHECK_RETURN(cudaEventRecord(start,0));
}

GpuCudaTimer::~GpuCudaTimer()
{
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(end));
}

float GpuCudaTimer::MillisecondsElapsed()
{
	float elapsed_time;
	CUDA_CHECK_RETURN(cudaEventRecord(end, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(end));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsed_time, start, end));
	return elapsed_time;
}

float GpuCudaTimer::SecondsElapsed()
{
	return 1000.0 * MillisecondsElapsed();
}

}


#endif
