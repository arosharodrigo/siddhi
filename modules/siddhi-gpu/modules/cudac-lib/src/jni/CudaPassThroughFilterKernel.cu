/*
 * CudaPassThroughFilterKernel.cu
 *
 *  Created on: Nov 9, 2014
 *      Author: prabodha
 */

#include "CudaPassThroughFilterKernel.h"
#include "CudaFilterKernelCore.h"

namespace SiddhiGpu
{

__global__ void ProcessEventsPassThroughFilterKernel(MultipleFilterKernelInput * _pInput, int * _pMatchedFilters)
{
	return;
}

CudaPassThroughFilterKernel::CudaPassThroughFilterKernel(GpuEventConsumer * _pConsumer, FILE * _fpLog) :
		CudaFilterKernel(0, _pConsumer, _fpLog)
{

}

CudaPassThroughFilterKernel::~CudaPassThroughFilterKernel()
{

}

void CudaPassThroughFilterKernel::Initialize()
{
	fprintf(fp_Log, "CudaPassThroughFilterKernel::Initialize \n");
}

void CudaPassThroughFilterKernel::ProcessEvents()
{

}

void CudaPassThroughFilterKernel::AddEvent(const CudaEvent * _pEvent)
{

}

void CudaPassThroughFilterKernel::AddAndProcessEvents(CudaEvent ** _apEvent, int _iEventCount)
{

}

void CudaPassThroughFilterKernel::AddFilterToDevice(Filter * _pFilter)
{

}

void CudaPassThroughFilterKernel::CopyFiltersToDevice()
{

}

};
