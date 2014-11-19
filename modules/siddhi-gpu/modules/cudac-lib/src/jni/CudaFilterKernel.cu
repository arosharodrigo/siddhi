/*
 * CudaFilterKernel.cu
 *
 *  Created on: Oct 17, 2014
 *      Author: prabodha
 */

#ifndef CUDAFILTERKERNEL_CU_
#define CUDAFILTERKERNEL_CU_

#include "GpuEventConsumer.h"
#include "CudaFilterKernel.h"
#include "Filter.h"
#include "CudaEvent.h"
#include "helper_timer.h"
#include "CudaFilterKernelCore.h"

namespace SiddhiGpu
{

// entry kernel
__global__ void ProcessEventsKernel(MultipleFilterKernelInput * _pInput, int * _pMatchedFilters)
{
//	__shared__ Filter mFilter;

//	if(threadIdx.x == 0 && threadIdx.y == 0)
//	{
		// shared mem initialize

//		mFilter = _pInput->ap_Filters[blockIdx.x];
//
//		mExecutors[EXECUTOR_NOOP] = NoopOperator;
//
//		mExecutors[EXECUTOR_AND] = AndCondition;
//		mExecutors[EXECUTOR_OR] = OrCondition;
//		mExecutors[EXECUTOR_NOT] = NotCondition;
//		mExecutors[EXECUTOR_BOOL] = BooleanCondition;
//
//		mExecutors[EXECUTOR_EQ_BOOL_BOOL] = EqualCompareBoolBool;
//		mExecutors[EXECUTOR_EQ_INT_INT] = EqualCompareIntInt;
//		mExecutors[EXECUTOR_EQ_LONG_LONG] = EqualCompareLongLong;
//		mExecutors[EXECUTOR_EQ_FLOAT_FLOAT] = EqualCompareFloatFloat;
//		mExecutors[EXECUTOR_EQ_DOUBLE_DOUBLE] = EqualCompareDoubleDouble;
//		mExecutors[EXECUTOR_EQ_STRING_STRING] = EqualCompareStringString;
//
//		mExecutors[EXECUTOR_NE_BOOL_BOOL] = NotEqualCompareBoolBool;
//		mExecutors[EXECUTOR_NE_INT_INT] = NotEqualCompareIntInt;
//		mExecutors[EXECUTOR_NE_LONG_LONG] = NotEqualCompareLongLong;
//		mExecutors[EXECUTOR_NE_FLOAT_FLOAT] = NotEqualCompareFloatFloat;
//		mExecutors[EXECUTOR_NE_DOUBLE_DOUBLE] = NotEqualCompareDoubleDouble;
//		mExecutors[EXECUTOR_NE_STRING_STRING] = NotEqualCompareStringString;
//
//		mExecutors[EXECUTOR_GT_INT_INT] = GreaterThanCompareIntInt;
//		mExecutors[EXECUTOR_GT_LONG_LONG] = GreaterThanCompareLongLong;
//		mExecutors[EXECUTOR_GT_FLOAT_FLOAT] = GreaterThanCompareFloatFloat;
//		mExecutors[EXECUTOR_GT_DOUBLE_DOUBLE] = GreaterThanCompareDoubleDouble;
//
//		mExecutors[EXECUTOR_LT_INT_INT] = LessThanCompareIntInt;
//		mExecutors[EXECUTOR_LT_LONG_LONG] = LessThanCompareLongLong;
//		mExecutors[EXECUTOR_LT_FLOAT_FLOAT] = LessThanCompareFloatFloat;
//		mExecutors[EXECUTOR_LT_DOUBLE_DOUBLE] = LessThanCompareDoubleDouble;
//
//		mExecutors[EXECUTOR_GE_INT_INT] = GreaterAndEqualCompareIntInt;
//		mExecutors[EXECUTOR_GE_LONG_LONG] = GreaterAndEqualCompareLongLong;
//		mExecutors[EXECUTOR_GE_FLOAT_FLOAT] = GreaterAndEqualCompareFloatFloat;
//		mExecutors[EXECUTOR_GE_DOUBLE_DOUBLE] = GreaterAndEqualCompareDoubleDouble;
//
//		mExecutors[EXECUTOR_LE_INT_INT] = LessAndEqualCompareIntInt;
//		mExecutors[EXECUTOR_LE_LONG_LONG] = LessAndEqualCompareLongLong;
//		mExecutors[EXECUTOR_LE_FLOAT_FLOAT] = LessAndEqualCompareFloatFloat;
//		mExecutors[EXECUTOR_LE_DOUBLE_DOUBLE] = LessAndEqualCompareDoubleDouble;
//
//		mExecutors[EXECUTOR_INVALID] = InvalidOperator;
//
//		__syncthreads();
//	}

	if(threadIdx.x >= _pInput->i_EventCount || blockIdx.x >= _pInput->i_FilterCount || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	// get assigned event
	CudaEvent mEvent = _pInput->ap_EventBuffer[threadIdx.x];

//	// get assigned filter
	Filter mFilter = _pInput->ap_Filters[blockIdx.x];

	int iCurrentNodeIdx = 0;
	bool bResult = Evaluate(mEvent, mFilter, iCurrentNodeIdx);

//	bool bResult = true;

	if(bResult)
	{
		int iFilterCountIdx = (mFilter.i_FilterId * _pInput->i_MaxEventCount) + threadIdx.x;
		_pMatchedFilters[iFilterCountIdx] = 1;
	}
	else // ~ possible way to avoid cudaMemset from host
	{
		int iFilterCountIdx = (mFilter.i_FilterId * _pInput->i_MaxEventCount) + threadIdx.x;
		_pMatchedFilters[iFilterCountIdx] = 0;
	}
}

// #######################################################################################################

CudaFilterKernel::CudaFilterKernel(int _iMaxBufferSize, GpuEventConsumer * _pConsumer, FILE * _fpLog) :
		CudaKernelBase(_pConsumer, _fpLog),
		i_MaxEventBufferSize(_iMaxBufferSize),
		i_NumEvents(0)
{
	ap_HostEventBuffer = NULL;
	p_HostInput= NULL;
	p_DeviceInput = NULL;
	p_StopWatch = NULL;
	i_NumAttributes = 0;
	i_FilterCount = 0;
	pi_DeviceMatchedEvents = NULL;
	pi_HostMachedEvents = NULL;
}

CudaFilterKernel::~CudaFilterKernel()
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

void CudaFilterKernel::OnExit()
{
	CUDA_CHECK_RETURN(cudaDeviceReset());
	CUDA_CHECK_RETURN(cudaProfilerStop());
}

void CudaFilterKernel::Initialize()
{
	fprintf(fp_Log, "CudaFilterKernel::Initialize \n");

	sdkCreateTimer(&p_StopWatch);

	p_HostInput = (MultipleFilterKernelInput*) malloc(sizeof(MultipleFilterKernelInput));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &p_DeviceInput, sizeof(MultipleFilterKernelInput)));

	CUDA_CHECK_RETURN(cudaMallocHost((void**) &ap_HostEventBuffer, sizeof(CudaEvent) * i_MaxEventBufferSize));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &p_HostInput->ap_EventBuffer, sizeof(CudaEvent) * i_MaxEventBufferSize)); // will be cudaMemCpy later
	p_HostInput->i_MaxEventCount = i_MaxEventBufferSize;
}

void CudaFilterKernel::ProcessEvents()
{
	sdkStartTimer(&p_StopWatch);

	p_HostInput->i_EventCount = i_NumEvents;

//	fprintf(_fp, "[ProcessEvents] MemcpyHostToDevice EventBuffer : EventCount=%d/%d/%d MemSize=%lu \n",
//			p_HostInput->i_EventCount, i_NumEvents, i_MaxEventBufferSize, sizeof(CudaEvent) * i_NumEvents);
//	fprintf(_fp, "[ProcessEvents] MemcpyHostToDevice EventBuffer : DevicePtr=%p HostPtr=%p \n",
//			p_HostInput->ap_EventBuffer, ap_HostEventBuffer);

	//TODO: async copy
	CUDA_CHECK_RETURN(cudaMemcpy(p_HostInput->ap_EventBuffer, ap_HostEventBuffer, sizeof(CudaEvent) * i_MaxEventBufferSize, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(p_DeviceInput, p_HostInput, sizeof(MultipleFilterKernelInput), cudaMemcpyHostToDevice));

//	CUDA_CHECK_RETURN(cudaMemset(pi_DeviceMatchedEvents, 0, sizeof(int) * i_MaxEventBufferSize * i_FilterCount)); // TODO: do this in 0th thread

	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	// call entry kernel
	int numBlocksX = i_FilterCount;
	int numBlocksY = 1;
	dim3 numBlocks = dim3(numBlocksX, numBlocksY);
	dim3 numThreads = dim3(i_NumEvents, 1);

	ProcessEventsKernel<<<numBlocks, numThreads>>>(p_DeviceInput, pi_DeviceMatchedEvents);
	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	fprintf(fp_Log, "[ProcessEvents] Copying back results\n");
	CUDA_CHECK_RETURN(cudaMemcpy(
			pi_HostMachedEvents,
			pi_DeviceMatchedEvents,
			sizeof(int) * i_MaxEventBufferSize * i_FilterCount,
			cudaMemcpyDeviceToHost));

	sdkStopTimer(&p_StopWatch);

	// process results
	fprintf(fp_Log, "[ProcessEvents] Results are : \n");
	for(int i=0; i<i_FilterCount; ++i)
	{
		for(int j=0; j<i_NumEvents; ++j)
		{
			if(pi_HostMachedEvents[i * i_MaxEventBufferSize + j])
			{
				p_EventConsumer->OnCudaEventMatch(j, i);
				fprintf(fp_Log, "\t Event %d matched to Filter %d\n", j, i);
			}
		}
	}

	float fElapsed = sdkGetTimerValue(&p_StopWatch);
	fprintf(fp_Log, "[ProcessEvents] Stats : Elapsed=%f ms\n", fElapsed);
	fflush(fp_Log);

	lst_ElapsedTimes.push_back(fElapsed);

	sdkResetTimer(&p_StopWatch);
	i_NumEvents = 0;
}

void CudaFilterKernel::AddEvent(const CudaEvent * _pEvent)
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

void CudaFilterKernel::AddAndProcessEvents(CudaEvent ** _apEvent, int _iEventCount)
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

void CudaFilterKernel::AddFilterToDevice(Filter * _pFilter)
{
	lst_HostFilters.push_back(_pFilter);
}

void CudaFilterKernel::CopyFiltersToDevice()
{
	i_FilterCount  = lst_HostFilters.size();

	CUDA_CHECK_RETURN(cudaMalloc(
			(void**) &p_HostInput->ap_Filters,
			sizeof(Filter) * i_FilterCount));

	Filter * apHostFilters = (Filter *) malloc(sizeof(Filter) * i_FilterCount);

	std::list<Filter*>::iterator ite = lst_HostFilters.begin();
	for(int i=0; i<i_FilterCount; ++i, ite++)
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
			p_HostInput->ap_Filters,
			apHostFilters,
			sizeof(Filter) * i_FilterCount,
			cudaMemcpyHostToDevice));

	pi_HostMachedEvents = (int*) malloc(sizeof(int) * i_MaxEventBufferSize * i_FilterCount);
	CUDA_CHECK_RETURN(cudaMalloc((void**) &pi_DeviceMatchedEvents, sizeof(int) * i_MaxEventBufferSize * i_FilterCount));

	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	p_HostInput->i_FilterCount = i_FilterCount;

	free(apHostFilters);
	apHostFilters = NULL;

	lst_HostFilters.clear();

}

float CudaFilterKernel::GetElapsedTimeAverage()
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

#endif /* CUDAFILTERKERNEL_CU_ */
