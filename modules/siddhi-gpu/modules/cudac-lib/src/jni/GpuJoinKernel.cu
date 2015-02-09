#ifndef _GPU_JOIN_KERNEL_CU__
#define _GPU_JOIN_KERNEL_CU__

#include <stdio.h>
#include <stdlib.h>
#include "GpuMetaEvent.h"
#include "GpuProcessor.h"
#include "GpuProcessorContext.h"
#include "GpuStreamEventBuffer.h"
#include "GpuRawByteBuffer.h"
#include "GpuIntBuffer.h"
#include "GpuKernelDataTypes.h"
#include "GpuJoinProcessor.h"
#include "GpuJoinKernel.h"
#include "GpuCudaHelper.h"

namespace SiddhiGpu
{

__global__
void ProcessEventsJoin(
		char               * _pInputEventBuffer,     // original input events buffer
		GpuKernelMetaEvent * _pMetaEvent,            // Meta event for original input events
		int                  _iNumberOfEvents,       // Number of events in input buffer (matched + not matched)
		char               * _pEventWindowBuffer,    // Event window buffer
		int                  _iWindowLength,         // Length of current events window
		int                  _iRemainingCount,       // Remaining free slots in Window buffer
		char               * _pResultsBuffer,        // Resulting events buffer
		int                  _iMaxEventCount,        // used for setting results array
		int                  _iSizeOfEvent,          // Size of an event
		int                  _iEventsPerBlock        // number of events allocated per block
)
{
	// avoid out of bound threads
	if(threadIdx.x >= _iEventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _iNumberOfEvents / _iEventsPerBlock) && // last thread block
			(threadIdx.x >= _iNumberOfEvents % _iEventsPerBlock)) // extra threads
	{
		return;
	}

	// get assigned event
	int iEventIdx = (blockIdx.x * _iEventsPerBlock) + threadIdx.x;

	// get in event starting position
	char * pInEventBuffer = _pInputEventBuffer + (_iSizeOfEvent * iEventIdx);

	// output to results buffer [expired event, in event]
	char * pResultsExpiredEventBuffer = _pResultsBuffer + (_iSizeOfEvent * iEventIdx * 2);
	char * pResultsInEventBuffer = pResultsExpiredEventBuffer + _iSizeOfEvent;

	GpuEvent * pExpiredEvent = (GpuEvent *)pResultsExpiredEventBuffer;
	// calculate in/expired event pair for this event

	if(iEventIdx >= _iRemainingCount)
	{
		if(iEventIdx < _iWindowLength)
		{
			// in window buffer
			char * pExpiredOutEventInWindowBuffer = _pEventWindowBuffer + (_iSizeOfEvent * (iEventIdx - _iRemainingCount));

			GpuEvent * pWindowEvent = (GpuEvent*) pExpiredOutEventInWindowBuffer;
			if(pWindowEvent->i_Type != GpuEvent::NONE) // if window event is filled
			{
				memcpy(pResultsExpiredEventBuffer, pExpiredOutEventInWindowBuffer, _iSizeOfEvent);
				pExpiredEvent->i_Type = GpuEvent::EXPIRED;
			}
			else
			{
				memset(pResultsExpiredEventBuffer, 0, _iSizeOfEvent);
				pExpiredEvent->i_Type = GpuEvent::NONE;
			}
		}
		else
		{
			// in input event buffer
			char * pExpiredOutEventInInputBuffer = _pInputEventBuffer + (_iSizeOfEvent * (iEventIdx - _iWindowLength));

			memcpy(pResultsExpiredEventBuffer, pExpiredOutEventInInputBuffer, _iSizeOfEvent);
			pExpiredEvent->i_Type = GpuEvent::EXPIRED;
		}
	}
	else
	{
		// [NULL,inEvent]
		memset(pResultsExpiredEventBuffer, 0, _iSizeOfEvent);
		pExpiredEvent->i_Type = GpuEvent::NONE;

	}

	memcpy(pResultsInEventBuffer, pInEventBuffer, _iSizeOfEvent);

}

__global__
void JoinSetWindowState(
		char               * _pInputEventBuffer,     // original input events buffer
		int                  _iNumberOfEvents,       // Number of events in input buffer (matched + not matched)
		char               * _pEventWindowBuffer,    // Event window buffer
		int                  _iWindowLength,         // Length of current events window
		int                  _iRemainingCount,       // Remaining free slots in Window buffer
		int                  _iMaxEventCount,        // used for setting results array
		int                  _iSizeOfEvent,          // Size of an event
		int                  _iEventsPerBlock        // number of events allocated per block
)
{
	// avoid out of bound threads
	if(threadIdx.x >= _iEventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _iNumberOfEvents / _iEventsPerBlock) && // last thread block
			(threadIdx.x >= _iNumberOfEvents % _iEventsPerBlock)) // extra threads
	{
		return;
	}

	// get assigned event
	int iEventIdx = (blockIdx.x * _iEventsPerBlock) + threadIdx.x;

	// get in event starting position
	char * pInEventBuffer = _pInputEventBuffer + (_iSizeOfEvent * iEventIdx);

	if(_iNumberOfEvents < _iWindowLength)
	{
		int iWindowPositionShift = _iWindowLength - _iNumberOfEvents;

		if(_iRemainingCount < _iNumberOfEvents)
		{
			int iExitEventCount = _iNumberOfEvents - _iRemainingCount;

			// calculate start and end window buffer positions
			int iStart = iEventIdx + iWindowPositionShift;
			int iEnd = iStart;
			while(iEnd >= 0)
			{
				char * pDestinationEventBuffer = _pEventWindowBuffer + (_iSizeOfEvent * iEnd);
				GpuEvent * pDestinationEvent = (GpuEvent*) pDestinationEventBuffer;

				if(pDestinationEvent->i_Type != GpuEvent::NONE) // there is an event in destination position
				{
					iEnd -= iExitEventCount;
				}
				else
				{
					break;
				}

			}

			// work back from end while copying events
			while(iEnd < iStart)
			{
				char * pDestinationEventBuffer = _pEventWindowBuffer + (_iSizeOfEvent * iEnd);
				GpuEvent * pDestinationEvent = (GpuEvent*) pDestinationEventBuffer;

				char * pSourceEventBuffer = _pEventWindowBuffer + (_iSizeOfEvent * (iEnd + iExitEventCount));

				memcpy(pDestinationEventBuffer, pSourceEventBuffer, _iSizeOfEvent);
				pDestinationEvent->i_Type = GpuEvent::EXPIRED;

				iEnd += iExitEventCount;
			}

			// iEnd == iStart
			if(iStart >= 0)
			{
				char * pDestinationEventBuffer = _pEventWindowBuffer + (_iSizeOfEvent * iStart);
				GpuEvent * pDestinationEvent = (GpuEvent*) pDestinationEventBuffer;
				memcpy(pDestinationEventBuffer, pInEventBuffer, _iSizeOfEvent);
				pDestinationEvent->i_Type = GpuEvent::EXPIRED;
			}
		}
		else
		{
			// just copy event to window
			iWindowPositionShift -= (_iRemainingCount - _iNumberOfEvents);

			char * pWindowEventBuffer = _pEventWindowBuffer + (_iSizeOfEvent * (iEventIdx + iWindowPositionShift));

			memcpy(pWindowEventBuffer, pInEventBuffer, _iSizeOfEvent);
			GpuEvent * pExpiredEvent = (GpuEvent*) pWindowEventBuffer;
			pExpiredEvent->i_Type = GpuEvent::EXPIRED;
		}
	}
	else
	{
		int iWindowPositionShift = _iNumberOfEvents - _iWindowLength;

		if(iEventIdx >= iWindowPositionShift)
		{
			char * pWindowEventBuffer = _pEventWindowBuffer + (_iSizeOfEvent * (iEventIdx - iWindowPositionShift));

			memcpy(pWindowEventBuffer, pInEventBuffer, _iSizeOfEvent);
			GpuEvent * pExpiredEvent = (GpuEvent*) pWindowEventBuffer;
			pExpiredEvent->i_Type = GpuEvent::EXPIRED;
		}
	}
}


// ======================================================================================================================

GpuJoinKernel::GpuJoinKernel(GpuProcessor * _pProc, GpuProcessorContext * _pLeftContext, GpuProcessorContext * _pRightContext,
		int _iThreadBlockSize, int _iLeftWindowSize, int _iRightWindowSize, FILE * _fpLeftLog, FILE * _fpRightLog) :
	GpuKernel(_pProc, _pLeftContext->GetDeviceId(), _iThreadBlockSize, _fpLeftLog),
	p_LeftContext(_pLeftContext),
	p_RightContext(_pRightContext),
	i_LeftInputBufferIndex(0),
	i_RightInputBufferIndex(0),
	p_LeftInputEventBuffer(NULL),
	p_RightInputEventBuffer(NULL),
	p_LeftWindowEventBuffer(NULL),
	p_RightWindowEventBuffer(NULL),
	p_ResultEventBuffer(NULL),
	i_LeftStreamWindowSize(_iLeftWindowSize),
	i_RightStreamWindowSize(_iRightWindowSize),
	i_LeftRemainingCount(_iLeftWindowSize),
	i_RightRemainingCount(_iRightWindowSize),
	b_DeviceSet(false),
	i_InitializedStreamCount(0),
	fp_LeftLog(_fpLeftLog),
	fp_RightLog(_fpRightLog)
{
	p_JoinProcessor = (GpuJoinProcessor*) _pProc;
}

GpuJoinKernel::~GpuJoinKernel()
{
	fprintf(fp_LeftLog, "[GpuJoinKernel] destroy\n");
	fflush(fp_LeftLog);
	fprintf(fp_RightLog, "[GpuJoinKernel] destroy\n");
	fflush(fp_RightLog);
}

bool GpuJoinKernel::Initialize(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize)
{
	fprintf(fp_LeftLog, "[GpuJoinKernel] Initialize : StreamIndex=%d\n", _iStreamIndex);
	fflush(fp_LeftLog);
	fprintf(fp_RightLog, "[GpuJoinKernel] Initialize : StreamIndex=%d\n", _iStreamIndex);
	fflush(fp_RightLog);

	if(_iStreamIndex == 0)
	{
		// set input event buffer
		fprintf(fp_LeftLog, "[GpuJoinKernel] Left InpuEventBufferIndex=%d\n", i_LeftInputBufferIndex);
		fflush(fp_LeftLog);
		p_LeftInputEventBuffer = (GpuStreamEventBuffer*) p_LeftContext->GetEventBuffer(i_LeftInputBufferIndex);
		p_LeftInputEventBuffer->Print();

		// left event window

		p_LeftWindowEventBuffer = new GpuStreamEventBuffer("LeftWindowEventBuffer", p_LeftContext->GetDeviceId(), _pMetaEvent, fp_LeftLog);
		p_LeftWindowEventBuffer->CreateEventBuffer(i_LeftStreamWindowSize);

		fprintf(fp_LeftLog, "[GpuJoinKernel] Created device left window buffer : Length=%d Size=%d bytes\n", i_LeftStreamWindowSize,
				p_LeftWindowEventBuffer->GetEventBufferSizeInBytes());
		fflush(fp_LeftLog);

		fprintf(fp_LeftLog, "[GpuJoinKernel] initialize left window buffer data \n");
		fflush(fp_LeftLog);
		p_LeftWindowEventBuffer->Print();

		p_LeftWindowEventBuffer->ResetHostEventBuffer(0);

		char * pLeftHostWindowBuffer = p_LeftWindowEventBuffer->GetHostEventBuffer();
		char * pCurrentEvent;
		for(int i=0; i<i_LeftStreamWindowSize; ++i)
		{
			pCurrentEvent = pLeftHostWindowBuffer + (_pMetaEvent->i_SizeOfEventInBytes * i);
			GpuEvent * pGpuEvent = (GpuEvent*) pCurrentEvent;
			pGpuEvent->i_Type = GpuEvent::NONE;
		}
		p_LeftWindowEventBuffer->CopyToDevice(false);

		i_InitializedStreamCount++;
	}
	else if(_iStreamIndex == 1)
	{
		fprintf(fp_RightLog, "[GpuJoinKernel] Right InpuEventBufferIndex=%d\n", i_RightInputBufferIndex);
		fflush(fp_RightLog);
		p_RightInputEventBuffer = (GpuStreamEventBuffer*) p_RightContext->GetEventBuffer(i_RightInputBufferIndex);
		p_RightInputEventBuffer->Print();

		// right event window

		p_RightWindowEventBuffer = new GpuStreamEventBuffer("RightWindowEventBuffer", p_RightContext->GetDeviceId(), _pMetaEvent, fp_RightLog);
		p_RightWindowEventBuffer->CreateEventBuffer(i_RightStreamWindowSize);

		fprintf(fp_RightLog, "[GpuJoinKernel] Created device right window buffer : Length=%d Size=%d bytes\n", i_RightStreamWindowSize,
				p_RightWindowEventBuffer->GetEventBufferSizeInBytes());
		fflush(fp_RightLog);

		fprintf(fp_RightLog, "[GpuJoinKernel] initialize right window buffer data \n");
		fflush(fp_RightLog);
		p_RightWindowEventBuffer->Print();

		p_RightWindowEventBuffer->ResetHostEventBuffer(0);

		char * pRightHostWindowBuffer = p_RightWindowEventBuffer->GetHostEventBuffer();
		char * pCurrentEvent;
		for(int i=0; i<i_RightStreamWindowSize; ++i)
		{
			pCurrentEvent = pRightHostWindowBuffer + (_pMetaEvent->i_SizeOfEventInBytes * i);
			GpuEvent * pGpuEvent = (GpuEvent*) pCurrentEvent;
			pGpuEvent->i_Type = GpuEvent::NONE;
		}
		p_RightWindowEventBuffer->CopyToDevice(false);

		i_InitializedStreamCount++;
	}

	if(i_InitializedStreamCount == 2)
	{
		if(p_ResultEventBuffer == NULL)
		{
			// set resulting event buffer and its meta data
			int iResultBufferSizeInBytes = 0;
			if(p_JoinProcessor->GetLeftTrigger())
			{
				iResultBufferSizeInBytes += (p_LeftInputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes +
						(p_RightInputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes * i_RightStreamWindowSize)) * 2
								*  p_LeftInputEventBuffer->GetMaxEventCount();
			}
			if(p_JoinProcessor->GetRightTrigger())
			{
				iResultBufferSizeInBytes += (p_RightInputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes +
						(p_LeftInputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes * i_LeftStreamWindowSize)) * 2
								*  p_RightInputEventBuffer->GetMaxEventCount();
			}

			p_ResultEventBuffer = new GpuRawByteBuffer("JoinResultEventBuffer", p_LeftContext->GetDeviceId(), fp_LeftLog);
			p_ResultEventBuffer->CreateEventBuffer(iResultBufferSizeInBytes);

			fprintf(fp_LeftLog, "[GpuJoinKernel] ResultEventBuffer created : Size=%d bytes\n", p_ResultEventBuffer->GetEventBufferSizeInBytes());
			fflush(fp_LeftLog);
			fprintf(fp_RightLog, "[GpuJoinKernel] ResultEventBuffer created : Size=%d bytes\n", p_ResultEventBuffer->GetEventBufferSizeInBytes());
			fflush(fp_RightLog);
			p_ResultEventBuffer->Print();

		}

		fprintf(fp_LeftLog, "[GpuJoinKernel] Initialization complete\n");
		fflush(fp_LeftLog);
		fprintf(fp_RightLog, "[GpuJoinKernel] Initialization complete\n");
		fflush(fp_RightLog);
	}

	return true;
}

void GpuJoinKernel::Process(int _iStreamIndex, int & _iNumEvents, bool _bLast)
{
#ifdef GPU_DEBUG
	FILE * fpLog = fp_Log;
	if(_iStreamIndex == 0)
	{
		fpLog = fp_LeftLog;
	}
	else if (_iStreamIndex == 1)
	{
		fpLog = fp_RightLog;
	}
	fprintf(fpLog, "[GpuJoinKernel] Process : StreamIndex=%d EventCount=%d\n", _iStreamIndex, _iNumEvents);
	fflush(fpLog);
#endif

	if(!b_DeviceSet)
	{
		if(_iStreamIndex == 0)
		{
			GpuCudaHelper::SelectDevice(i_DeviceId, "GpuJoinKernel", fp_LeftLog);
		}
		else if (_iStreamIndex == 1)
		{
			GpuCudaHelper::SelectDevice(i_DeviceId, "GpuJoinKernel", fp_RightLog);
		}
		b_DeviceSet = true;
	}

#ifdef KERNEL_TIME
	sdkStartTimer(&p_StopWatch);
#endif

	// call entry kernel
	int numBlocksX = ceil((float)_iNumEvents / (float)i_ThreadBlockSize);
	int numBlocksY = 1;
	dim3 numBlocks = dim3(numBlocksX, numBlocksY);
	dim3 numThreads = dim3(i_ThreadBlockSize, 1);

#ifdef GPU_DEBUG
	fprintf(fpLog, "[GpuJoinKernel] Invoke kernel Blocks(%d,%d) Threads(%d,%d)\n", numBlocksX, numBlocksY, i_ThreadBlockSize, 1);
	fflush(fpLog);
#endif

//	ProcessEventsJoin<<<numBlocks, numThreads>>>(
//			p_InputEventBuffer->GetDeviceEventBuffer(),
//			p_InputEventBuffer->GetDeviceMetaEvent(),
//			_iNumEvents,
//			p_WindowEventBuffer->GetDeviceEventBuffer(),
//			i_WindowSize,
//			i_RemainingCount,
//			p_ResultEventBuffer->GetDeviceEventBuffer(),
//			p_InputEventBuffer->GetMaxEventCount(),
//			p_InputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes,
//			i_ThreadBlockSize
//	);
//
//	JoinSetWindowState<<<numBlocks, numThreads>>>(
//			p_InputEventBuffer->GetDeviceEventBuffer(),
//			_iNumEvents,
//			p_WindowEventBuffer->GetDeviceEventBuffer(),
//			i_WindowSize,
//			i_RemainingCount,
//			p_InputEventBuffer->GetMaxEventCount(),
//			p_InputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes,
//			i_ThreadBlockSize
//	);

	if(_bLast)
	{
		p_ResultEventBuffer->CopyToHost(true);
	}

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

#ifdef GPU_DEBUG
	fprintf(fpLog, "[GpuJoinKernel] Kernel complete \n");
	fflush(fpLog);
#endif

#ifdef GPU_DEBUG
	fprintf(fpLog, "[GpuJoinKernel] Results copied \n");
	fflush(fpLog);
#endif

#ifdef KERNEL_TIME
	sdkStopTimer(&p_StopWatch);
	float fElapsed = sdkGetTimerValue(&p_StopWatch);
	fprintf(fpLog, "[GpuJoinKernel] Stats : Elapsed=%f ms\n", fElapsed);
	fflush(fpLog);
	lst_ElapsedTimes.push_back(fElapsed);
	sdkResetTimer(&p_StopWatch);
#endif

	if(_iStreamIndex == 0)
	{
		if(_iNumEvents > i_LeftRemainingCount)
		{
			i_LeftRemainingCount = 0;
		}
		else
		{
			i_LeftRemainingCount -= _iNumEvents;
		}
	}
	else if(_iStreamIndex == 1)
	{
		if(_iNumEvents > i_RightRemainingCount)
		{
			i_RightRemainingCount = 0;
		}
		else
		{
			i_RightRemainingCount -= _iNumEvents;
		}
	}

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
