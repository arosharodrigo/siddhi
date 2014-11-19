/*
 * CudaSingleFilterKernel.cu
 *
 *  Created on: Nov 9, 2014
 *      Author: prabodha
 */

#ifndef CUDASINGLEFILTERKERNEL_CU_
#define CUDASINGLEFILTERKERNEL_CU_

#include "GpuEventConsumer.h"
#include "CudaSingleFilterKernel.h"
#include "Filter.h"
#include "CudaEvent.h"
#include "helper_timer.h"
#include "CudaFilterKernelCore.h"

namespace SiddhiGpu
{

__global__ void ProcessEventsSingleFilterKernel(SingleFilterKernelInput * _pInput, int * _pMatchedFilters)
{
	if(threadIdx.x >= _pInput->i_EventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _pInput->i_EventCount / _pInput->i_EventsPerBlock) && // last thread block
			(threadIdx.x >= _pInput->i_EventCount % _pInput->i_EventsPerBlock))
	{
		return;
	}

	// get assigned event
	int iEventIdx = (blockIdx.x * _pInput->i_EventsPerBlock) +  threadIdx.x;
	CudaEvent mEvent = _pInput->ap_EventBuffer[iEventIdx];

	// get assigned filter
	Filter mFilter = *_pInput->ap_Filter;

	int iCurrentNodeIdx = 0;
	bool bResult = Evaluate(mEvent, mFilter, iCurrentNodeIdx);

	if(bResult)
	{
		_pMatchedFilters[iEventIdx] = 1;
	}
	else // ~ possible way to avoid cudaMemset from host
	{
		_pMatchedFilters[iEventIdx] = 0;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CudaSingleFilterKernel::CudaSingleFilterKernel(int _iMaxBufferSize, GpuEventConsumer * _pConsumer, FILE * _fpLog) :
		CudaKernelBase(_pConsumer, _fpLog),
		i_MaxEventBufferSize(_iMaxBufferSize),
		i_NumEvents(0)
{
	i_EventsPerBlock = _iMaxBufferSize / 4; // TODO: change this dynamically based on MaxBuffersize

	ap_HostEventBuffer = NULL;
	p_HostInput= NULL;
	p_DeviceInput = NULL;
	p_StopWatch = NULL;
	i_NumAttributes = 0;
	pi_DeviceMatchedEvents = NULL;
	pi_HostMachedEvents = NULL;
}

CudaSingleFilterKernel::CudaSingleFilterKernel(int _iMaxBufferSize, int _iEventsPerBlock, GpuEventConsumer * _pConsumer, FILE * _fpLog) :
	CudaKernelBase(_pConsumer, _fpLog),
	i_MaxEventBufferSize(_iMaxBufferSize),
	i_NumEvents(0)
{
	if(_iEventsPerBlock > 0)
	{
		i_EventsPerBlock = _iEventsPerBlock;
	}
	else
	{
		i_EventsPerBlock = _iMaxBufferSize / 4;
	}

	ap_HostEventBuffer = NULL;
	p_HostInput= NULL;
	p_DeviceInput = NULL;
	p_StopWatch = NULL;
	i_NumAttributes = 0;
	pi_DeviceMatchedEvents = NULL;
	pi_HostMachedEvents = NULL;
}

CudaSingleFilterKernel::~CudaSingleFilterKernel()
{
	CUDA_CHECK_RETURN(cudaFree(p_DeviceInput->ap_EventBuffer));
	CUDA_CHECK_RETURN(cudaFree(p_DeviceInput));

	CUDA_CHECK_RETURN(cudaFreeHost(ap_HostEventBuffer));
	free(p_HostInput);

	CUDA_CHECK_RETURN(cudaDeviceReset());

	CUDA_CHECK_RETURN(cudaFree(pi_DeviceMatchedEvents));
	free(pi_HostMachedEvents);

	sdkDeleteTimer(&p_StopWatch);
}

void CudaSingleFilterKernel::Initialize()
{
	fprintf(fp_Log, "CudaSingleFilterKernel::Initialize \n");

	sdkCreateTimer(&p_StopWatch);

	p_HostInput = (SingleFilterKernelInput*) malloc(sizeof(SingleFilterKernelInput));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &p_DeviceInput, sizeof(SingleFilterKernelInput)));

	CUDA_CHECK_RETURN(cudaMallocHost((void**) &ap_HostEventBuffer, sizeof(CudaEvent) * i_MaxEventBufferSize));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &p_HostInput->ap_EventBuffer, sizeof(CudaEvent) * i_MaxEventBufferSize)); // will be cudaMemCpy later
	p_HostInput->i_MaxEventCount = i_MaxEventBufferSize;
}

void CudaSingleFilterKernel::ProcessEvents()
{
	sdkStartTimer(&p_StopWatch);

	p_HostInput->i_EventCount = i_NumEvents;

	//TODO: async copy
	CUDA_CHECK_RETURN(cudaMemcpy(p_HostInput->ap_EventBuffer, ap_HostEventBuffer, sizeof(CudaEvent) * i_MaxEventBufferSize, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(p_DeviceInput, p_HostInput, sizeof(SingleFilterKernelInput), cudaMemcpyHostToDevice));

//	CUDA_CHECK_RETURN(cudaMemset(pi_DeviceMatchedEvents, 0, sizeof(int) * i_MaxEventBufferSize * i_FilterCount)); // TODO: do this in 0th thread

	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	// call entry kernel
	int numBlocksX = i_NumEvents / i_EventsPerBlock;
	int numBlocksY = 1;
	dim3 numBlocks = dim3(numBlocksX, numBlocksY);
	dim3 numThreads = dim3(i_EventsPerBlock, 1);

	ProcessEventsSingleFilterKernel<<<numBlocks, numThreads>>>(p_DeviceInput, pi_DeviceMatchedEvents);
	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	//fprintf(fp_Log, "[ProcessEvents] Copying back results\n");
	CUDA_CHECK_RETURN(cudaMemcpy(
			pi_HostMachedEvents,
			pi_DeviceMatchedEvents,
			sizeof(int) * i_MaxEventBufferSize,
			cudaMemcpyDeviceToHost));

	sdkStopTimer(&p_StopWatch);

	// process results
	//fprintf(fp_Log, "[ProcessEvents] Results are : \n");
	int iCount = 0;
	for(int j=0; j<i_NumEvents; ++j) // TODO: can use bit map to reduce copy size. Is that efficient??
	{
		if(pi_HostMachedEvents[j])
		{
			p_EventConsumer->OnCudaEventMatch(j, 0);
			//fprintf(fp_Log, "\t Event %d matched to Filter %d\n", j, 0);
			iCount++;
		}
	}


	float fElapsed = sdkGetTimerValue(&p_StopWatch);
	fprintf(fp_Log, "[ProcessEvents] Stats : Matched=%d Elapsed=%f ms\n", iCount, fElapsed);
	fflush(fp_Log);

	lst_ElapsedTimes.push_back(fElapsed);

	sdkResetTimer(&p_StopWatch);
	i_NumEvents = 0;
}

void CudaSingleFilterKernel::AddEvent(const CudaEvent * _pEvent)
{
	// can just do a memcpy for whole structure
	ap_HostEventBuffer[i_NumEvents].ui_NumAttributes = _pEvent->ui_NumAttributes;

	for(int i=0; i<_pEvent->ui_NumAttributes; ++i)
	{
		ap_HostEventBuffer[i_NumEvents].a_Attributes[i].e_Type = _pEvent->a_Attributes[i].e_Type;
		ap_HostEventBuffer[i_NumEvents].a_Attributes[i].m_Value = _pEvent->a_Attributes[i].m_Value;
	}

	//printf("CudaKernel::AddEvent : EventIndex=%d\n", i_NumEvents);

	i_NumEvents++;
}

void CudaSingleFilterKernel::AddAndProcessEvents(CudaEvent ** _apEvent, int _iEventCount)
{
	for(int i=0; i<_iEventCount; ++i)
	{
		// can just do a memcpy for whole structure
		ap_HostEventBuffer[i].ui_NumAttributes = _apEvent[i]->ui_NumAttributes;

		for(int j=0; j<_apEvent[i]->ui_NumAttributes; ++j)
		{
			ap_HostEventBuffer[i].a_Attributes[j].e_Type = _apEvent[i]->a_Attributes[j].e_Type;
			ap_HostEventBuffer[i].a_Attributes[j].m_Value = _apEvent[i]->a_Attributes[j].m_Value;
		}

//		printf("CudaKernel::AddAndProcessEvents : EventIndex=%d\n", i_NumEvents);
	}
	i_NumEvents = _iEventCount;
	ProcessEvents();
}

void CudaSingleFilterKernel::AddFilterToDevice(Filter * _pFilter)
{
	lst_HostFilters.push_back(_pFilter);
}

void CudaSingleFilterKernel::CopyFiltersToDevice()
{
	if(lst_HostFilters.size() > 1)
	{
		fprintf(fp_Log, "[ERROR] More than one filter defined in CudaSingleFilterKernel : FilterCount=%lu", lst_HostFilters.size());
		fprintf(fp_Log, "[ERROR] Using the first filter for processing");
	}

	CUDA_CHECK_RETURN(cudaMalloc(
			(void**) &p_HostInput->ap_Filter,
			sizeof(Filter)));

	Filter * apHostFilters = (Filter *) malloc(sizeof(Filter));

	std::list<Filter*>::iterator ite = lst_HostFilters.begin();
	for(int i=0; i<1; ++i, ite++)
	{
		Filter * pFilter = *ite;

		apHostFilters[i].i_FilterId = pFilter->i_FilterId;
		apHostFilters[i].i_NodeCount = pFilter->i_NodeCount;
		apHostFilters[i].ap_ExecutorNodes = NULL;

		CUDA_CHECK_RETURN(cudaMalloc(
				(void**) &apHostFilters[i].ap_ExecutorNodes,
				sizeof(ExecutorNode) * pFilter->i_NodeCount));

		CUDA_CHECK_RETURN(cudaMemcpy(
				apHostFilters[i].ap_ExecutorNodes,
				pFilter->ap_ExecutorNodes,
				sizeof(ExecutorNode) * pFilter->i_NodeCount,
				cudaMemcpyHostToDevice));

		delete pFilter;
	}

	CUDA_CHECK_RETURN(cudaMemcpy(
			p_HostInput->ap_Filter,
			apHostFilters,
			sizeof(Filter),
			cudaMemcpyHostToDevice));

	pi_HostMachedEvents = (int*) malloc(sizeof(int) * i_MaxEventBufferSize);
	CUDA_CHECK_RETURN(cudaMalloc((void**) &pi_DeviceMatchedEvents, sizeof(int) * i_MaxEventBufferSize));

	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	p_HostInput->i_EventsPerBlock = i_EventsPerBlock;

	free(apHostFilters);
	apHostFilters = NULL;

	lst_HostFilters.clear();

}

float CudaSingleFilterKernel::GetElapsedTimeAverage()
{
	float total = 0;
	std::list<float>::iterator ite = lst_ElapsedTimes.begin();
	while(ite != lst_ElapsedTimes.end())
	{
		total += *ite;
		++ite;
	}

	return (total / lst_ElapsedTimes.size());
}

};

#endif


