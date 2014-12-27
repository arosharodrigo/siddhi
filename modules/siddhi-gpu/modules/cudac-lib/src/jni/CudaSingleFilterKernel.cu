/*
 * CudaSingleFilterKernel.cu
 *
 *  Created on: Nov 9, 2014
 *      Author: prabodha
 */

#ifndef CUDASINGLEFILTERKERNEL_CU_
#define CUDASINGLEFILTERKERNEL_CU_

#include "GpuEventConsumer.h"
#include "ByteBufferStructs.h"
#include "CudaSingleFilterKernel.h"
#include "Filter.h"
#include "CudaCommon.h"
#include "helper_timer.h"
#include "CudaFilterKernelCore.h"

namespace SiddhiGpu
{

__global__ void ProcessEventsSingleFilterKernel(SingleFilterKernelInput * _pInput)
{
	if(threadIdx.x >= _pInput->i_EventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _pInput->i_EventCount / _pInput->i_EventsPerBlock) && // last thread block
			(threadIdx.x >= _pInput->i_EventCount % _pInput->i_EventsPerBlock))
	{
		return;
	}

	EventMeta * pEventMeta = (EventMeta*) (_pInput->p_ByteBuffer + _pInput->i_EventMetaPosition);
	/*__shared__*/ EventMeta mEventMeta = *pEventMeta;

	// get assigned event
	int iEventIdx = (blockIdx.x * _pInput->i_EventsPerBlock) +  threadIdx.x;
	char * pEvent = (_pInput->p_ByteBuffer + _pInput->i_EventDataPosition) + (_pInput->i_SizeOfEvent * iEventIdx);

	// get assigned filter
	/*__shared__*/ Filter mFilter = *_pInput->ap_Filter;

	// get results array
	MatchedEvents * pMatchedEvents = (MatchedEvents*) (_pInput->p_ByteBuffer + _pInput->i_ResultsPosition);

	int iCurrentNodeIdx = 0;
	bool bResult = Evaluate(mFilter, mEventMeta, pEvent, iCurrentNodeIdx);

	//TODO improve results sending
	if(bResult)
	{
		pMatchedEvents->a_ResultEvents[iEventIdx] = 1;
	}
	else // ~ possible way to avoid cudaMemset from host
	{
		pMatchedEvents->a_ResultEvents[iEventIdx] = 0;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CudaSingleFilterKernel::CudaSingleFilterKernel(int _iMaxBufferSize, GpuEventConsumer * _pConsumer, FILE * _fpLog) :
		CudaKernelBase(_pConsumer, _fpLog),
		i_MaxNumberOfEvents(_iMaxBufferSize)
{
	i_EventsPerBlock = _iMaxBufferSize / 4; // TODO: change this dynamically based on MaxBuffersize

	p_HostEventBuffer = NULL;
	i_EventBufferSize = 0;
	p_HostInput= NULL;
	p_DeviceInput = NULL;
	p_StopWatch = NULL;
	i_NumAttributes = 0;
	i_CudaDeviceId = -1;
}

CudaSingleFilterKernel::CudaSingleFilterKernel(int _iMaxBufferSize, int _iEventsPerBlock, GpuEventConsumer * _pConsumer, FILE * _fpLog) :
	CudaKernelBase(_pConsumer, _fpLog),
	i_MaxNumberOfEvents(_iMaxBufferSize)
{
	if(_iEventsPerBlock > 0)
	{
		i_EventsPerBlock = _iEventsPerBlock;
	}
	else
	{
		i_EventsPerBlock = _iMaxBufferSize / 4;
	}

	p_HostEventBuffer = NULL;
	i_EventBufferSize = 0;
	p_HostInput= NULL;
	p_DeviceInput = NULL;
	p_StopWatch = NULL;
	i_NumAttributes = 0;
	i_CudaDeviceId = -1;
}

CudaSingleFilterKernel::~CudaSingleFilterKernel()
{
	fprintf(fp_Log, "CudaSingleFilterKernel destroy\n");
	fflush(fp_Log);

	CUDA_CHECK_RETURN(cudaFree(p_DeviceInput->p_ByteBuffer));
	CUDA_CHECK_RETURN(cudaFree(p_DeviceInput));
	p_DeviceInput = NULL;

	free(p_HostInput);
	p_HostInput = NULL;

	CUDA_CHECK_RETURN(cudaDeviceReset());

	sdkDeleteTimer(&p_StopWatch);
	p_StopWatch = NULL;
}

bool CudaSingleFilterKernel::Initialize(int _iCudaDeviceId)
{
	fprintf(fp_Log, "CudaSingleFilterKernel::Initializing with device Id %d [EventsPerBlock=%d]\n", _iCudaDeviceId, i_EventsPerBlock);

	if(SelectDevice(_iCudaDeviceId))
	{
		sdkCreateTimer(&p_StopWatch);

		p_HostInput = (SingleFilterKernelInput*) malloc(sizeof(SingleFilterKernelInput)); // host allocate Kernel input struct
		CUDA_CHECK_RETURN(cudaMalloc((void**) &p_DeviceInput, sizeof(SingleFilterKernelInput))); // device allocate Kernel input struct

		p_HostInput->i_MaxEventCount = i_MaxNumberOfEvents;

		p_HostInput->i_EventsPerBlock = i_EventsPerBlock;

		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		CUDA_CHECK_RETURN(cudaThreadSynchronize());

		return true;
	}
	else
	{
		return false;
	}

}

bool CudaSingleFilterKernel::SelectDevice(int _iDeviceId)
{
	fprintf(fp_Log, "Selecting CUDA device\n");

	int iDevCount = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&iDevCount));

	fprintf(fp_Log, "CUDA device count : %d\n", iDevCount);

	if(iDevCount == 0)
	{
		fprintf(fp_Log, "No CUDA devices found\n");
		fflush(fp_Log);
		return false;
	}

	if(_iDeviceId < iDevCount)
	{
		i_CudaDeviceId = _iDeviceId;

		CUDA_CHECK_RETURN(cudaSetDevice(i_CudaDeviceId));
		fprintf(fp_Log, "CUDA device set to %d\n", i_CudaDeviceId);
		fflush(fp_Log);
		return true;
	}

	fprintf(fp_Log, "CUDA device id %d is wrong\n", _iDeviceId);
	fflush(fp_Log);
	return false;

//	cudaDeviceProp mDeviceProp;
//	size_t iTotalMem, iFreeMem;
//	for(int i=0; i<iDevCount; ++i)
//	{
//		CUDA_CHECK_RETURN(cudaGetDeviceProperties(&mDeviceProp, i));
//		CUDA_CHECK_RETURN(cudaMemGetInfo(&iFreeMem, &iTotalMem));
//		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
//
//		printf("GPU device %d - %s (%d.%d) - Memory=[%d/%d] \n", i, mDeviceProp.name, mDeviceProp.major, mDeviceProp.minor,
//				iFreeMem, iTotalMem);
//
//	}

}

void CudaSingleFilterKernel::SetEventBuffer(char * _pBuffer, int _iSize)
{
	p_HostEventBuffer = _pBuffer;
	i_EventBufferSize = _iSize;

	p_HostInput->i_ResultsPosition = i_ResultsBufferPosition;
	p_HostInput->i_EventMetaPosition = i_EventMetaBufferPosition;
	p_HostInput->i_EventDataPosition = i_EventDataBufferPosition;
	p_HostInput->i_SizeOfEvent = i_SizeOfEvent;
	
	fprintf(fp_Log, "CudaSingleFilterKernel Allocating ByteBuffer in GPU : %d \n", (int)(sizeof(char) * i_EventBufferSize));
	fflush(fp_Log);

	CUDA_CHECK_RETURN(cudaMalloc((void**) &p_HostInput->p_ByteBuffer, sizeof(char) * i_EventBufferSize)); // device allocate ByteBuffer
	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	fprintf(fp_Log, "CudaSingleFilterKernel EventBuffer [Ptr=%p Size=%d]\n", p_HostEventBuffer, i_EventBufferSize);
	fprintf(fp_Log, "CudaSingleFilterKernel ResultsBufferPosition   : %d\n", i_ResultsBufferPosition);
	fprintf(fp_Log, "CudaSingleFilterKernel EventMetaBufferPosition : %d\n", i_EventMetaBufferPosition);
	fprintf(fp_Log, "CudaSingleFilterKernel EventDataBufferPosition : %d\n", i_EventDataBufferPosition);
	fprintf(fp_Log, "CudaSingleFilterKernel SizeOfEvent             : %d\n", i_SizeOfEvent);
	fprintf(fp_Log, "Device byte buffer ptr : %p \n", p_HostInput->p_ByteBuffer);
	fflush(fp_Log);
}

void CudaSingleFilterKernel::ProcessEvents(int _iNumEvents)
{
	CUDA_CHECK_RETURN(cudaSetDevice(i_CudaDeviceId)); // TODO: do this only at start

	sdkStartTimer(&p_StopWatch);

	p_HostInput->i_EventCount = _iNumEvents;


//	fprintf(fp_Log, "Device byte buffer ptr : %p \n", p_HostInput->p_ByteBuffer);

	//TODO: async copy
	CUDA_CHECK_RETURN(cudaMemcpy(p_HostInput->p_ByteBuffer, p_HostEventBuffer, sizeof(char) * i_EventBufferSize, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(p_DeviceInput, p_HostInput, sizeof(SingleFilterKernelInput), cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	// call entry kernel
	int numBlocksX = _iNumEvents / i_EventsPerBlock;
	int numBlocksY = 1;
	dim3 numBlocks = dim3(numBlocksX, numBlocksY);
	dim3 numThreads = dim3(i_EventsPerBlock, 1);

//	fprintf(fp_Log, "Invoke kernel Blocks(%d,%d) Threads(%d,%d)\n", numBlocksX, numBlocksY, i_EventsPerBlock, 1);
//	fprintf(fp_Log, "DeviceInput : %p \n", p_DeviceInput);
//	fprintf(fp_Log, "DeviceInput : EventCount=%d MaxCount=%d PerBlockCount=%d ResultsPosition=%d EventMetaPosition=%d"
//			" EventDataPosition=%d SizeOfEvent=%d ByteBuffer=%p Filter=%p \n", p_HostInput->i_EventCount,
//			p_HostInput->i_MaxEventCount, p_HostInput->i_EventsPerBlock,
//			p_HostInput->i_ResultsPosition, p_HostInput->i_EventMetaPosition, p_HostInput->i_EventDataPosition,
//			p_HostInput->i_SizeOfEvent, p_HostInput->p_ByteBuffer, p_HostInput->ap_Filter);
//	fflush(fp_Log);

	ProcessEventsSingleFilterKernel<<<numBlocks, numThreads>>>(p_DeviceInput);
	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

//	fprintf(fp_Log, "Kernel complete \n");
//	fflush(fp_Log);

	CUDA_CHECK_RETURN(cudaMemcpy(
			p_HostEventBuffer,
			p_HostInput->p_ByteBuffer,
			sizeof(char) * 4 * i_MaxNumberOfEvents,
			cudaMemcpyDeviceToHost));

//	fprintf(fp_Log, "Results copied \n");
//	fflush(fp_Log);

	sdkStopTimer(&p_StopWatch);

	float fElapsed = sdkGetTimerValue(&p_StopWatch);
	//fprintf(fp_Log, "[ProcessEvents] Stats : Elapsed=%f ms\n", fElapsed);
	//fflush(fp_Log);

	lst_ElapsedTimes.push_back(fElapsed);

	sdkResetTimer(&p_StopWatch);
//	i_NumEvents = 0;
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
		fflush(fp_Log);
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


	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

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


