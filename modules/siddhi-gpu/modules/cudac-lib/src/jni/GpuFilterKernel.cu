#ifndef _GPU_FILTER_KERNEL_CU__
#define _GPU_FILTER_KERNEL_CU__

#include <stdio.h>
#include <stdlib.h>
#include "GpuStreamEventBuffer.h"
#include "GpuIntBuffer.h"
#include "GpuMetaEvent.h"
#include "GpuProcessor.h"
#include "GpuProcessorContext.h"
#include "GpuCudaHelper.h"
#include "GpuUtils.h"
#include "GpuFilterProcessor.h"
#include "GpuKernelDataTypes.h"
#include "GpuFilterKernelCore.h"
#include "GpuFilterKernel.h"

#include <cub/cub.cuh>

namespace SiddhiGpu
{

// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds
#define THREADS_PER_BLOCK 256

#if __CUDA_ARCH__ >= 200
#define MY_KERNEL_MAX_THREADS (2 * THREADS_PER_BLOCK)
#define MY_KERNEL_MIN_BLOCKS 3
#else
#define MY_KERNEL_MAX_THREADS THREADS_PER_BLOCK
#define MY_KERNEL_MIN_BLOCKS 2
#endif


__global__
//__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
void ProcessEventsFilterKernelStandalone(
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

__global__
//__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
void ProcessEventsFilterKernelFirst(
		char               * _pInByteBuffer,      // Input ByteBuffer from java side
		GpuKernelMetaEvent * _pMetaEvent,         // Meta event of input events
		GpuKernelFilter    * _apFilter,           // Filters buffer - pre-copied at initialization
		int                  _iEventCount,        // Num events in this batch
		int                  _iMaxEventCount,     // used for setting results array
		int                  _iSizeOfEvent,       // Size of an event
		int                  _iEventsPerBlock,    // number of events allocated per block
		int                * _pMatchedIndexBuffer,// Matched event index buffer
		int                * _iMatchedCount       // matched event count
)
{
	__shared__ int iSharedCounter;

	if(threadIdx.x >= _iEventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _iEventCount / _iEventsPerBlock) && // last thread block
			(threadIdx.x >= _iEventCount % _iEventsPerBlock))
	{
		return;
	}

	if (threadIdx.x == 0)
	{
		iSharedCounter = 0;
	}
	__syncthreads();

	// get assigned event
	int iEventIdx = (blockIdx.x * _iEventsPerBlock) + threadIdx.x;

	char * pInEventBuffer = _pInByteBuffer + (_iSizeOfEvent * iEventIdx);

	// get assigned filter
	/*__shared__*/ GpuKernelFilter mFilter = *_apFilter;

	int iCurrentNodeIdx = 0;
	bool bMatched = Evaluate(mFilter, _pMetaEvent, pInEventBuffer, iCurrentNodeIdx);

	int iPositionInBlock;

	if(bMatched)
	{
		iPositionInBlock = atomicAdd(&iSharedCounter, 1);
	}
	__syncthreads();

	if(threadIdx.x == 0)
	{
		iSharedCounter = atomicAdd(_iMatchedCount, iSharedCounter);
	}
	__syncthreads();

	if(bMatched)
	{
		iPositionInBlock += iSharedCounter; // increment local pos by global counter
		_pMatchedIndexBuffer[iPositionInBlock] = iEventIdx;
//		_pResultBuffer[iEventIdx] = iEventIdx;
//		_pResultBuffer[atomicAdd(_iMatchedCount, 1)] = iEventIdx;
	}
//	else // ~ possible way to avoid cudaMemset from host
//	{
//		_pResultBuffer[iEventIdx] = -1 * iEventIdx;
//	}
}

__global__
//__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
void ProcessEventsFilterKernelFirstV2(
		char               * _pInByteBuffer,      // Input ByteBuffer from java side
		GpuKernelMetaEvent * _pMetaEvent,         // Meta event of input events
		GpuKernelFilter    * _apFilter,           // Filters buffer - pre-copied at initialization
		int                  _iEventCount,        // Num events in this batch
		int                  _iMaxEventCount,     // used for setting results array
		int                  _iSizeOfEvent,       // Size of an event
		int                  _iEventsPerBlock,    // number of events allocated per block
		int                * _pMatchedIndexBuffer // Matched event index buffer
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

	char * pInEventBuffer = _pInByteBuffer + (_iSizeOfEvent * iEventIdx);

	// get assigned filter
	/*__shared__*/ GpuKernelFilter mFilter = *_apFilter;

	int iCurrentNodeIdx = 0;
	bool bMatched = Evaluate(mFilter, _pMetaEvent, pInEventBuffer, iCurrentNodeIdx);

	if(bMatched)
	{
		_pMatchedIndexBuffer[iEventIdx] = 1;
	}
	else // ~ possible way to avoid cudaMemset from host
	{
		_pMatchedIndexBuffer[iEventIdx] = 0;
	}
}

__global__
//__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
void ResultSorter(
		char               * _pInByteBuffer,      // Input ByteBuffer from java side
		int                * _pMatchedIndexBuffer,// Matched event index buffer
		int                * _pPrefixSumBuffer,   // prefix sum buffer
		int                  _iEventCount,        // Num events in original batch
		int                  _iSizeOfEvent,       // Size of an event
		int                  _iEventsPerBlock,    // number of events allocated per block
		char               * _pOutputEventBuffer  // Matched events final buffer
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

	if(_pMatchedIndexBuffer[iEventIdx] == 0)
	{
		return;
	}

	char * pInEventBuffer = _pInByteBuffer + (_iSizeOfEvent * iEventIdx);
	char * pOutEventBuffer = _pOutputEventBuffer + (_iSizeOfEvent * (_pPrefixSumBuffer[iEventIdx]));

	memcpy(pOutEventBuffer, pInEventBuffer, _iSizeOfEvent);

}

// ============================================================================================================

GpuFilterKernelStandalone::GpuFilterKernelStandalone(GpuProcessor * _pProc, GpuProcessorContext * _pContext, int _iThreadBlockSize, FILE * _fPLog) :
	GpuKernel(_pProc, _pContext->GetDeviceId(), _iThreadBlockSize, _fPLog),
	p_Context(_pContext),
	p_DeviceFilter(NULL),
	p_InputEventBuffer(NULL),
	p_ResultEventBuffer(NULL),
	b_DeviceSet(false)
{

}

GpuFilterKernelStandalone::~GpuFilterKernelStandalone()
{
	fprintf(fp_Log, "[GpuFilterKernelStandalone] destroy\n");
	fflush(fp_Log);

	CUDA_CHECK_RETURN(cudaFree(p_DeviceFilter));
	p_DeviceFilter = NULL;

//	sdkDeleteTimer(&p_StopWatch);
//	p_StopWatch = NULL;
}

bool GpuFilterKernelStandalone::Initialize(GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize)
{
	fprintf(fp_Log, "[GpuFilterKernelStandalone] Initialize\n");
	fflush(fp_Log);

	// set input event buffer
	fprintf(fp_Log, "[GpuFilterKernelStandalone] InputEventBufferIndex=%d\n", i_InputBufferIndex);
	fflush(fp_Log);
	p_InputEventBuffer = (GpuStreamEventBuffer*)p_Context->GetEventBuffer(i_InputBufferIndex);
	p_InputEventBuffer->Print();

	// set resulting event buffer and its meta data
	GpuMetaEvent * pFilterResultMetaEvent = new GpuMetaEvent(_pMetaEvent->i_StreamIndex, 1, sizeof(int));
	pFilterResultMetaEvent->SetAttribute(0, DataType::Int, sizeof(int), 0);

	p_ResultEventBuffer = new GpuIntBuffer(p_Context->GetDeviceId(), pFilterResultMetaEvent, fp_Log);
	p_ResultEventBuffer->CreateEventBuffer(_iInputEventBufferSize);
	i_ResultEventBufferIndex = p_Context->AddEventBuffer(p_ResultEventBuffer);
	fprintf(fp_Log, "[GpuFilterKernelStandalone] ResultEventBuffer created : Index=%d Size=%d bytes\n", i_ResultEventBufferIndex,
			p_ResultEventBuffer->GetEventBufferSizeInBytes());
	fflush(fp_Log);
	p_ResultEventBuffer->Print();

	delete pFilterResultMetaEvent;


	fprintf(fp_Log, "[GpuFilterKernelStandalone] Copying filter to device \n");
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

	fprintf(fp_Log, "[GpuFilterKernelStandalone] Initialization complete \n");
	fflush(fp_Log);

	return true;

}

void GpuFilterKernelStandalone::Process(int & _iNumEvents, bool _bLast)
{
#ifdef GPU_DEBUG
	GpuUtils::PrintByteBuffer(p_InputEventBuffer->GetHostEventBuffer(), _iNumEvents, p_InputEventBuffer->GetHostMetaEvent(), "GpuFilterKernelStandalone", fp_Log);
#endif

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
	fprintf(fp_Log, "[GpuFilterKernelStandalone] Invoke kernel Blocks(%d,%d) Threads(%d,%d)\n", numBlocksX, numBlocksY, i_ThreadBlockSize, 1);
	fflush(fp_Log);
#endif

	ProcessEventsFilterKernelStandalone<<<numBlocks, numThreads>>>(
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

	fprintf(fp_Log, "[GpuFilterKernelStandalone] ProcessEventsFilterKernelStandalone results\n");
	int * pResults = p_ResultEventBuffer->GetHostEventBuffer();
	for(int i=0; i<_iNumEvents; ++i)
	{
		fprintf(fp_Log, "[GpuFilterKernelStandalone] Result [%d => %d] \n", i, *pResults);
		pResults++;
	}
	fflush(fp_Log);
#endif

#ifdef GPU_DEBUG
	fprintf(fp_Log, "[GpuFilterKernelStandalone] Kernel complete \n");
	fflush(fp_Log);
#endif

#ifdef KERNEL_TIME
	sdkStopTimer(&p_StopWatch);
	float fElapsed = sdkGetTimerValue(&p_StopWatch);
	fprintf(fp_Log, "[GpuFilterKernelStandalone] Stats : Elapsed=%f ms\n", fElapsed);
	fflush(fp_Log);
	lst_ElapsedTimes.push_back(fElapsed);
	sdkResetTimer(&p_StopWatch);
#endif
}

char * GpuFilterKernelStandalone::GetResultEventBuffer()
{
	return (char*)p_ResultEventBuffer->GetHostEventBuffer();
}

int GpuFilterKernelStandalone::GetResultEventBufferSize()
{
	return p_ResultEventBuffer->GetEventBufferSizeInBytes();
}

// ============================================================================================================

GpuFilterKernelFirst::GpuFilterKernelFirst(GpuProcessor * _pProc, GpuProcessorContext * _pContext, int _iThreadBlockSize, FILE * _fPLog) :
	GpuKernel(_pProc, _pContext->GetDeviceId(), _iThreadBlockSize, _fPLog),
	p_Context(_pContext),
	p_DeviceFilter(NULL),
	p_InputEventBuffer(NULL),
	p_MatchedIndexEventBuffer(NULL),
	p_PrefixSumBuffer(NULL),
	p_ResultEventBuffer(NULL),
	i_MatchedEvenBufferIndex(-1),
	p_TempStorageForPrefixSum(NULL),
	i_SizeOfTempStorageForPrefixSum(0),
	b_DeviceSet(false)
{

}

GpuFilterKernelFirst::~GpuFilterKernelFirst()
{
	fprintf(fp_Log, "[GpuFilterKernelFirst] destroy\n");
	fflush(fp_Log);

	CUDA_CHECK_RETURN(cudaFree(p_DeviceFilter));
	p_DeviceFilter = NULL;

	delete p_ResultEventBuffer;
	p_ResultEventBuffer = NULL;

	delete p_MatchedIndexEventBuffer;
	p_MatchedIndexEventBuffer = NULL;

	delete p_PrefixSumBuffer;
	p_PrefixSumBuffer = NULL;

	CUDA_CHECK_RETURN(cudaFree(p_TempStorageForPrefixSum));
	p_TempStorageForPrefixSum = NULL;

//	sdkDeleteTimer(&p_StopWatch);
//	p_StopWatch = NULL;
}

bool GpuFilterKernelFirst::Initialize(GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize)
{
	fprintf(fp_Log, "[GpuFilterKernelFirst] Initialize\n");
	fflush(fp_Log);

	// set input event buffer
	fprintf(fp_Log, "[GpuFilterKernelFirst] InputEventBufferIndex=%d\n", i_InputBufferIndex);
	fflush(fp_Log);
	p_InputEventBuffer = (GpuStreamEventBuffer*)p_Context->GetEventBuffer(i_InputBufferIndex);
	p_InputEventBuffer->Print();

	// set resulting event buffer and its meta data
	GpuMetaEvent * pMatchedResultIndexMetaEvent = new GpuMetaEvent(_pMetaEvent->i_StreamIndex, 1, sizeof(int));
	pMatchedResultIndexMetaEvent->SetAttribute(0, DataType::Int, sizeof(int), 0);

	p_MatchedIndexEventBuffer = new GpuIntBuffer(p_Context->GetDeviceId(), pMatchedResultIndexMetaEvent, fp_Log);
	p_MatchedIndexEventBuffer->CreateEventBuffer(_iInputEventBufferSize);
	fprintf(fp_Log, "[GpuFilterKernelFirst] MatchedIndexEventBuffer created : Size=%d bytes\n",
			p_MatchedIndexEventBuffer->GetEventBufferSizeInBytes());
	fflush(fp_Log);
	p_MatchedIndexEventBuffer->Print();

	p_PrefixSumBuffer = new GpuIntBuffer(p_Context->GetDeviceId(), pMatchedResultIndexMetaEvent, fp_Log);
	p_PrefixSumBuffer->CreateEventBuffer(_iInputEventBufferSize);
	p_PrefixSumBuffer->Print();

	delete pMatchedResultIndexMetaEvent;

	i_SizeOfTempStorageForPrefixSum = sizeof(int) * 2 * _iInputEventBufferSize;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&p_TempStorageForPrefixSum, i_SizeOfTempStorageForPrefixSum));


	p_ResultEventBuffer = new GpuStreamEventBuffer(p_Context->GetDeviceId(), _pMetaEvent, fp_Log);
	p_ResultEventBuffer->CreateEventBuffer(_iInputEventBufferSize);

	i_ResultEventBufferIndex = p_Context->AddEventBuffer(p_ResultEventBuffer);

	fprintf(fp_Log, "[GpuFilterKernelFirst] ResultEventBuffer created : Index=%d Size=%d bytes\n", i_ResultEventBufferIndex,
			p_ResultEventBuffer->GetEventBufferSizeInBytes());
	fflush(fp_Log);
	p_ResultEventBuffer->Print();

	fprintf(fp_Log, "[GpuFilterKernelFirst] Copying filter to device \n");
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

	fprintf(fp_Log, "[GpuFilterKernelFirst] Initialization complete \n");
	fflush(fp_Log);

	return true;

}

void GpuFilterKernelFirst::Process(int & _iNumEvents, bool _bLast)
{
#ifdef GPU_DEBUG
	GpuUtils::PrintByteBuffer(p_InputEventBuffer->GetHostEventBuffer(), _iNumEvents, p_InputEventBuffer->GetHostMetaEvent(), "GpuFilterKernelFirst::In", fp_Log);
#endif

	if(!b_DeviceSet)
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
	fprintf(fp_Log, "[GpuFilterKernelFirst] Invoke kernel Blocks(%d,%d) Threads(%d,%d)\n", numBlocksX, numBlocksY, i_ThreadBlockSize, 1);
	fflush(fp_Log);
#endif

//	char               * _pInByteBuffer,      // Input ByteBuffer from java side
//	GpuKernelMetaEvent * _pMetaEvent,         // Meta event of input events
//	GpuKernelFilter    * _apFilter,           // Filters buffer - pre-copied at initialization
//	int                  _iEventCount,        // Num events in this batch
//	int                  _iMaxEventCount,     // used for setting results array
//	int                  _iSizeOfEvent,       // Size of an event
//	int                  _iEventsPerBlock,    // number of events allocated per block
//	int                * _pMatchedIndexBuffer // Matched event index buffer

	ProcessEventsFilterKernelFirstV2<<<numBlocks, numThreads>>>(
			p_InputEventBuffer->GetDeviceEventBuffer(),
			p_InputEventBuffer->GetDeviceMetaEvent(),
			p_DeviceFilter,
			_iNumEvents,
			p_InputEventBuffer->GetMaxEventCount(),
			p_InputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes,
			i_ThreadBlockSize,
			p_MatchedIndexEventBuffer->GetDeviceEventBuffer()
	);

	CUDA_CHECK_RETURN(cub::DeviceScan::ExclusiveSum(
			p_TempStorageForPrefixSum, i_SizeOfTempStorageForPrefixSum,
			p_MatchedIndexEventBuffer->GetDeviceEventBuffer(),
			p_PrefixSumBuffer->GetDeviceEventBuffer(),
			_iNumEvents));


//	char               * _pInByteBuffer,      // Input ByteBuffer from java side
//	int                * _pMatchedIndexBuffer,// Matched event index buffer
//	int                * _pPrefixSumBuffer,   // prefix sum buffer
//	int                  _iEventCount,        // Num events in original batch
//	int                  _iSizeOfEvent,       // Size of an event
//	int                  _iEventsPerBlock,    // number of events allocated per block
//	char               * _pOutputEventBuffer  // Matched events final buffer

	ResultSorter<<<numBlocks, numThreads>>>(
			p_InputEventBuffer->GetDeviceEventBuffer(),
			p_MatchedIndexEventBuffer->GetDeviceEventBuffer(),
			p_PrefixSumBuffer->GetDeviceEventBuffer(),
			_iNumEvents,
			p_InputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes,
			i_ThreadBlockSize,
			p_ResultEventBuffer->GetDeviceEventBuffer()
	);

	if(_bLast)
	{
		p_ResultEventBuffer->CopyToHost(true);
	}

	p_PrefixSumBuffer->CopyToHost(true);

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	int * pPrefixSumResults = p_PrefixSumBuffer->GetHostEventBuffer();
	_iNumEvents = pPrefixSumResults[_iNumEvents - 1];

#ifdef GPU_DEBUG
	if(_bLast)
	{
		GpuUtils::PrintByteBuffer(p_ResultEventBuffer->GetHostEventBuffer(), _iNumEvents, p_ResultEventBuffer->GetHostMetaEvent(),
				"GpuFilterKernelFirst::Out", fp_Log);
	}

	fprintf(fp_Log, "[GpuFilterKernelFirst] Kernel complete : ResultCount=%d\n", _iNumEvents);
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

char * GpuFilterKernelFirst::GetResultEventBuffer()
{
	return (char*)p_ResultEventBuffer->GetHostEventBuffer();
}

int GpuFilterKernelFirst::GetResultEventBufferSize()
{
	return p_ResultEventBuffer->GetEventBufferSizeInBytes();
}

}


#endif
