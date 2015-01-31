#ifndef _GPU_FILTER_KERNEL_CU__
#define _GPU_FILTER_KERNEL_CU__

#include <stdio.h>
#include <stdlib.h>
#include "GpuFilterKernel.h"
#include "GpuFilterProcessor.h"
#include "GpuKernelDataTypes.h"
#include "GpuFilterKernelCore.h"

namespace SiddhiGpu
{

__global__ void ProcessEventsSingleFilterKernel(
		char               * _pInByteBuffer,      // Input ByteBuffer from java side
		GpuKernelFilter    * _apFilter,           // Filters buffer - pre-copied at initialization
		GpuKernelMetaEvent * _pMetaEvent,         // Meta event of input events
		int                  _iMaxEventCount,     // used for setting results array
		int                  _iSizeOfEvent,       // Size of an event
		int                  _iEventsPerBlock,    // number of events allocated per block
		int                  _iEventCount,        // Num events in this batch
		int                * _pResultBuffer       // Result event index array
)
{
	if(threadIdx.x >= _iEventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _iEventCount / _iEventsPerBlock) && // last thread block
			(threadIdx.x >= _iEventCount % _iEventsPerBlock))
	{
		return;
	}

	// get assigned event
	int iEventIdx = (blockIdx.x * _iEventsPerBlock) + threadIdx.x;

	char * pEvent = _pInByteBuffer + (_iSizeOfEvent * iEventIdx);

	// get assigned filter
	/*__shared__*/ GpuKernelFilter mFilter = *_apFilter;

	int iCurrentNodeIdx = 0;
	bool bResult = Evaluate(mFilter, _pMetaEvent, pEvent, iCurrentNodeIdx);

	//TODO improve results sending
	if(bResult)
	{
		_pResultBuffer[iEventIdx] = iEventIdx;
	}
	else // ~ possible way to avoid cudaMemset from host
	{
		_pResultBuffer[iEventIdx] = -1 * iEventIdx;
	}
}

// ============================================================================================================

GpuFilterKernel::GpuFilterKernel(GpuProcessor * _pProc, GpuProcessorContext * _pContext, int _iThreadBlockSize, FILE * _fPLog) :
	GpuKernel(_pProc, _pContext->GetDeviceId(), _iThreadBlockSize, _fPLog),
	p_Context(_pContext),
	p_DeviceFilter(NULL),
	p_InputEventBuffer(NULL),
	p_ResultEventBuffer(NULL),
	b_DeviceSet(false)
{

}

GpuFilterKernel::~GpuFilterKernel()
{
	fprintf(fp_Log, "[GpuFilterKernel] destroy\n");
	fflush(fp_Log);

	CUDA_CHECK_RETURN(cudaFree(p_DeviceFilter));
	p_DeviceFilter = NULL;

//	sdkDeleteTimer(&p_StopWatch);
//	p_StopWatch = NULL;
}

bool GpuFilterKernel::Initialize(GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize)
{
	fprintf(fp_Log, "[GpuFilterKernel] Initialize\n");
	fflush(fp_Log);

	// set input event buffer
	fprintf(fp_Log, "[GpuFilterKernel] InputEventBufferIndex=%d\n", i_InputBufferIndex);
	fflush(fp_Log);
	p_InputEventBuffer = p_Context->GetEventBuffer(i_InputBufferIndex);

	// set resulting event buffer and its meta data
	GpuMetaEvent * pFilterResultMetaEvent = new GpuMetaEvent(_pMetaEvent->i_StreamIndex, 1, sizeof(int));
	pFilterResultMetaEvent->SetAttribute(0, DataType::Int, sizeof(int), 0);

	p_ResultEventBuffer = new GpuIntBuffer(p_Context->GetDeviceId(), pFilterResultMetaEvent, fp_Log);
	p_ResultEventBuffer->CreateEventBuffer(_iInputEventBufferSize);
	i_ResultEventBufferIndex = p_Context->AddEventBuffer(p_ResultEventBuffer);
	fprintf(fp_Log, "[GpuFilterKernel] ResultEventBuffer created : Index=%d Size=%d bytes\n", i_ResultEventBufferIndex,
			p_ResultEventBuffer->GetEventBufferSizeInBytes());
	fflush(fp_Log);

	delete pFilterResultMetaEvent;


	fprintf(fp_Log, "[GpuFilterKernel] Copying filter to device \n");
	fflush(fp_Log);

	CUDA_CHECK_RETURN(cudaMalloc(
			(void**) &p_DeviceFilter,
			sizeof(GpuKernelFilter)));

	GpuKernelFilter * apHostFilters = (GpuKernelFilter *) malloc(sizeof(GpuKernelFilter));

	GpuFilterProcessor * pFilter = (GpuFilterProcessor*)p_Processor;

	apHostFilters->i_NodeCount = pFilter->i_NodeCount;
	apHostFilters->ap_ExecutorNodes = NULL;

	CUDA_CHECK_RETURN(cudaMalloc(
			(void**) &apHostFilters->ap_ExecutorNodes,
			sizeof(ExecutorNode) * pFilter->i_NodeCount));

	CUDA_CHECK_RETURN(cudaMemcpy(
			apHostFilters->ap_ExecutorNodes,
			pFilter->ap_ExecutorNodes,
			sizeof(ExecutorNode) * pFilter->i_NodeCount,
			cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMemcpy(
			p_DeviceFilter,
			apHostFilters,
			sizeof(GpuKernelFilter),
			cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	free(apHostFilters);
	apHostFilters = NULL;

	fprintf(fp_Log, "[GpuFilterKernel] Initialization complete \n");
	fflush(fp_Log);

	return true;

}

void GpuFilterKernel::Process(int _iNumEvents, bool _bLast)
{
	if(!b_DeviceSet) // TODO: check if this works in every conditions. How Java thread pool works with disrupter?
	{
		GpuCudaHelper::SelectDevice(i_DeviceId, fp_Log);
		b_DeviceSet = true;
	}

#ifdef KERNEL_TIME
	sdkStartTimer(&p_StopWatch);
#endif

	// copy byte buffer
	p_InputEventBuffer->CopyToDevice(true);

	// call entry kernel
	int numBlocksX = ceil((float)_iNumEvents / (float)i_ThreadBlockSize);
	int numBlocksY = 1;
	dim3 numBlocks = dim3(numBlocksX, numBlocksY);
	dim3 numThreads = dim3(i_ThreadBlockSize, 1);

#ifdef GPU_DEBUG
	fprintf(fp_Log, "[GpuFilterKernel] Invoke kernel Blocks(%d,%d) Threads(%d,%d)\n", numBlocksX, numBlocksY, i_ThreadBlockSize, 1);
	fflush(fp_Log);
#endif

	ProcessEventsSingleFilterKernel<<<numBlocks, numThreads>>>(
			p_InputEventBuffer->GetDeviceEventBuffer(),
			p_DeviceFilter,
			p_InputEventBuffer->GetDeviceMetaEvent(),
			p_InputEventBuffer->GetMaxEventCount(),
			p_InputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes,
			i_ThreadBlockSize,
			_iNumEvents,
			p_ResultEventBuffer->GetDeviceEventBuffer()
	);

	if(_bLast)
	{
		p_ResultEventBuffer->CopyToHost(true);
	}

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

#ifdef GPU_DEBUG
	fprintf(fp_Log, "[GpuFilterKernel] Kernel complete \n");
	fflush(fp_Log);
#endif

//	CUDA_CHECK_RETURN(cudaMemcpy(
//			p_HostEventBuffer,
//			p_HostInput->p_ByteBuffer,
//			sizeof(char) * 4 * i_MaxNumberOfEvents,
//			cudaMemcpyDeviceToHost));

#ifdef GPU_DEBUG
	fprintf(fp_Log, "Results copied \n");
	fflush(fp_Log);
#endif

#ifdef KERNEL_TIME
	sdkStopTimer(&p_StopWatch);
	float fElapsed = sdkGetTimerValue(&p_StopWatch);
	fprintf(fp_Log, "[ProcessEvents] Stats : Elapsed=%f ms\n", fElapsed);
	fflush(fp_Log);
	lst_ElapsedTimes.push_back(fElapsed);
	sdkResetTimer(&p_StopWatch);
#endif
}

char * GpuFilterKernel::GetResultEventBuffer()
{
	return (char*)p_ResultEventBuffer->GetHostEventBuffer();
}

int GpuFilterKernel::GetResultEventBufferSize()
{
	return p_ResultEventBuffer->GetEventBufferSizeInBytes();
}

}


#endif
