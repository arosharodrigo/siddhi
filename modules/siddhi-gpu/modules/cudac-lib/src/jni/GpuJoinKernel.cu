#ifndef _GPU_JOIN_KERNEL_CU__
#define _GPU_JOIN_KERNEL_CU__

#include <stdio.h>
#include <stdlib.h>
#include "GpuMetaEvent.h"
#include "GpuProcessor.h"
#include "GpuProcessorContext.h"
#include "GpuStreamEventBuffer.h"
#include "GpuIntBuffer.h"
#include "GpuKernelDataTypes.h"
#include "GpuJoinKernel.h"
#include "GpuCudaHelper.h"

namespace SiddhiGpu
{

GpuJoinKernel::GpuJoinKernel(GpuProcessor * _pProc, GpuProcessorContext * _pContext,
		int _iThreadBlockSize, int _iLeftWindowSize, int _iRightWindowSize, FILE * _fPLog) :
	GpuKernel(_pProc, _pContext->GetDeviceId(), _iThreadBlockSize, _fPLog),
	p_Context(_pContext),
	p_InputEventBuffer(NULL),
	p_ResultEventBuffer(NULL),
	b_DeviceSet(false),
	i_LeftStraemWindowSize(_iLeftWindowSize),
	i_RightStraemWindowSize(_iRightWindowSize),
	i_LeftRemainingCount(_iLeftWindowSize),
	i_RightRemainingCount(_iRightWindowSize)
{

}

GpuJoinKernel::~GpuJoinKernel()
{
	fprintf(fp_Log, "[GpuJoinKernel] destroy\n");
	fflush(fp_Log);
}

bool GpuJoinKernel::Initialize(GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize)
{
	fprintf(fp_Log, "[GpuJoinKernel] Initialize\n");
	fflush(fp_Log);

	// set input event buffer
	fprintf(fp_Log, "[GpuJoinKernel] InpuEventBufferIndex=%d\n", i_InputBufferIndex);
	fflush(fp_Log);
	p_InputEventBuffer = (GpuStreamEventBuffer*) p_Context->GetEventBuffer(i_InputBufferIndex);

	// set resulting event buffer and its meta data
	p_ResultEventBuffer = new GpuStreamEventBuffer("JoinResultEventBuffer", p_Context->GetDeviceId(), _pMetaEvent, fp_Log);
	p_ResultEventBuffer->CreateEventBuffer(_iInputEventBufferSize * 2);

	i_ResultEventBufferIndex = p_Context->AddEventBuffer(p_ResultEventBuffer);

	fprintf(fp_Log, "[GpuJoinKernel] ResultEventBuffer created : Index=%d Size=%d bytes\n", i_ResultEventBufferIndex,
			p_ResultEventBuffer->GetEventBufferSizeInBytes());
	fflush(fp_Log);

	fprintf(fp_Log, "[GpuJoinKernel] Initialization complete\n");
	fflush(fp_Log);

	return true;
}

void GpuJoinKernel::Process(int & _iNumEvents, bool _bLast)
{
	fprintf(fp_Log, "[GpuJoinKernel] Process : EventCount=%d\n", _iNumEvents);
	fflush(fp_Log);

/*	if(!b_DeviceSet) // TODO: check if this works in every conditions. How Java thread pool works with disrupter?
	{
		GpuCudaHelper::SelectDevice(i_DeviceId, fp_Log);
		b_DeviceSet = true;
	}

	p_InputEventBuffer->CopyToDevice(true);

#ifdef KERNEL_TIME
	sdkStartTimer(&p_StopWatch);
#endif

	// call entry kernel
	int numBlocksX = ceil((float)_iNumEvents / (float)i_ThreadBlockSize);
	int numBlocksY = 1;
	dim3 numBlocks = dim3(numBlocksX, numBlocksY);
	dim3 numThreads = dim3(i_ThreadBlockSize, 1);

#ifdef GPU_DEBUG
	fprintf(fp_Log, "[GpuJoinKernel] Invoke kernel Blocks(%d,%d) Threads(%d,%d)\n", numBlocksX, numBlocksY, i_ThreadBlockSize, 1);
	fflush(fp_Log);
#endif

	ProcessEventsLengthSlidingWindow<<<numBlocks, numThreads>>>(
			p_InputEventBuffer->GetDeviceEventBuffer(),
			p_InputEventBuffer->GetDeviceMetaEvent(),
			_iNumEvents,
			p_WindowEventBuffer->GetDeviceEventBuffer(),
			i_WindowSize,
			i_RemainingCount,
			p_ResultEventBuffer->GetDeviceEventBuffer(),
			p_InputEventBuffer->GetMaxEventCount(),
			p_InputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes,
			i_ThreadBlockSize
	);

	SetWindowState<<<numBlocks, numThreads>>>(
			p_InputEventBuffer->GetDeviceEventBuffer(),
			_iNumEvents,
			p_WindowEventBuffer->GetDeviceEventBuffer(),
			i_WindowSize,
			i_RemainingCount,
			p_InputEventBuffer->GetMaxEventCount(),
			p_InputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes,
			i_ThreadBlockSize
	);

	if(_bLast)
	{
		p_ResultEventBuffer->CopyToHost(true);
	}

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

#ifdef GPU_DEBUG
	fprintf(fp_Log, "[GpuJoinKernel] Kernel complete \n");
	fflush(fp_Log);
#endif

	//	CUDA_CHECK_RETURN(cudaMemcpy(
	//			p_HostEventBuffer,
	//			p_HostInput->p_ByteBuffer,
	//			sizeof(char) * 4 * i_MaxNumberOfEvents,
	//			cudaMemcpyDeviceToHost));

#ifdef GPU_DEBUG
	fprintf(fp_Log, "[GpuJoinKernel] Results copied \n");
	fflush(fp_Log);
#endif

#ifdef KERNEL_TIME
	sdkStopTimer(&p_StopWatch);
	float fElapsed = sdkGetTimerValue(&p_StopWatch);
	fprintf(fp_Log, "[GpuJoinKernel] Stats : Elapsed=%f ms\n", fElapsed);
	fflush(fp_Log);
	lst_ElapsedTimes.push_back(fElapsed);
	sdkResetTimer(&p_StopWatch);
#endif

	if(_iNumEvents > i_RemainingCount)
	{
		i_RemainingCount = 0;
	}
	else
	{
		i_RemainingCount -= _iNumEvents;
	}
	*/
}

char * GpuJoinKernel::GetResultEventBuffer()
{
	return p_ResultEventBuffer->GetHostEventBuffer();
}

int GpuJoinKernel::GetResultEventBufferSize()
{
	return p_ResultEventBuffer->GetEventBufferSizeInBytes();
}

}

#endif
