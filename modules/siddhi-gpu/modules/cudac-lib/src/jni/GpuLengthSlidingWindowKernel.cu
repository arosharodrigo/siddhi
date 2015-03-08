#ifndef _GPU_LENGTH_SLIDING_WINDOW_KERNEL_CU__
#define _GPU_LENGTH_SLIDING_WINDOW_KERNEL_CU__

#include <stdio.h>
#include <stdlib.h>

#include "GpuMetaEvent.h"
#include "GpuProcessor.h"
#include "GpuProcessorContext.h"
#include "GpuStreamEventBuffer.h"
#include "GpuIntBuffer.h"
#include "GpuKernelDataTypes.h"
#include "GpuLengthSlidingWindowKernel.h"
#include "GpuCudaHelper.h"
#include "GpuUtils.h"

namespace SiddhiGpu
{

__global__
void ProcessEventsLengthSlidingWindow(
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

	memcpy(pResultsInEventBuffer, pInEventBuffer, _iSizeOfEvent);

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

}

 // not used
//__global__
//void ProcessEventsLengthSlidingWindowFilter(
//		char               * _pInputEventBuffer,     // original input events buffer
//		GpuKernelMetaEvent * _pMetaEvent,            // Meta event for original input events
//		int                  _iNumberOfEvents,       // Number of events in input buffer (matched + not matched)
//		int                * _pFilterdEventsIndexes, // Matched event indexes from filter kernel
//		char               * _pEventWindowBuffer,    // Event window buffer
//		int                  _iWindowLength,         // Length of current events window
//		int                  _iRemainingCount,       // Remaining free slots in Window buffer
//		char               * _pResultsBuffer,        // Resulting events buffer
//		int                  _iMaxEventCount,        // used for setting results array
//		int                  _iSizeOfEvent,          // Size of an event
//		int                  _iEventsPerBlock        // number of events allocated per block
//)
//{
//	// avoid out of bound threads
//	if(threadIdx.x >= _iEventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
//		return;
//
//	if((blockIdx.x == _iNumberOfEvents / _iEventsPerBlock) && // last thread block
//			(threadIdx.x >= _iNumberOfEvents % _iEventsPerBlock)) // extra threads
//	{
//		return;
//	}
//
//	// get assigned event
//	int iEventIdx = (blockIdx.x * _iEventsPerBlock) + threadIdx.x;
//
//	// get in event starting position
//	char * pInEventBuffer = _pInputEventBuffer + (_iSizeOfEvent * iEventIdx);
//
//	// output to results buffer [expired event, in event]
//	char * pResultsExpiredEventBuffer = _pResultsBuffer + (_iSizeOfEvent * iEventIdx * 2);
//	char * pResultsInEventBuffer = pResultsExpiredEventBuffer + _iSizeOfEvent;
//
//	GpuEvent * pExpiredEvent = (GpuEvent *)pResultsExpiredEventBuffer;
//
//	// check if event in my index is a matched event
//	if(_pFilterdEventsIndexes[iEventIdx] < 0)
//	{
//		memset(pResultsExpiredEventBuffer, 0, _iSizeOfEvent);
//		pExpiredEvent->i_Type = GpuEvent::NONE;
//
//		memset(pResultsInEventBuffer, 0, _iSizeOfEvent);
//		GpuEvent * pResultsInEvent = (GpuEvent *)pResultsInEventBuffer;
//		pResultsInEvent->i_Type = GpuEvent::NONE;
//
//		return; // not matched
//	}
//
//	// calculate in/expired event pair for this event
//
//
//	memcpy(pResultsInEventBuffer, pInEventBuffer, _iSizeOfEvent);
//
//	if(iEventIdx >= _iRemainingCount)
//	{
//		if(iEventIdx < _iWindowLength)
//		{
//			// in window buffer
//			char * pExpiredOutEventInWindowBuffer = _pEventWindowBuffer + (_iSizeOfEvent * (iEventIdx - _iRemainingCount));
//
//			GpuEvent * pWindowEvent = (GpuEvent*) pExpiredOutEventInWindowBuffer;
//			if(pWindowEvent->i_Type != GpuEvent::NONE) // if window event is filled
//			{
//				memcpy(pResultsExpiredEventBuffer, pExpiredOutEventInWindowBuffer, _iSizeOfEvent);
//				pExpiredEvent->i_Type = GpuEvent::EXPIRED;
//			}
//			else
//			{
//				memset(pResultsExpiredEventBuffer, 0, _iSizeOfEvent);
//				pExpiredEvent->i_Type = GpuEvent::NONE;
//			}
//		}
//		else
//		{
//			// in input event buffer
//			char * pExpiredOutEventInInputBuffer = _pInputEventBuffer + (_iSizeOfEvent * (iEventIdx - _iWindowLength));
//
//			memcpy(pResultsExpiredEventBuffer, pExpiredOutEventInInputBuffer, _iSizeOfEvent);
//			pExpiredEvent->i_Type = GpuEvent::EXPIRED;
//		}
//	}
//	else
//	{
//		// [NULL,inEvent]
//		memset(pResultsExpiredEventBuffer, 0, _iSizeOfEvent);
//		pExpiredEvent->i_Type = GpuEvent::NONE;
//
//	}
//
//}

__global__
void SetWindowState(
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
			int iPrevToEnd = iEnd;
			while(iEnd >= 0)
			{
				char * pDestinationEventBuffer = _pEventWindowBuffer + (_iSizeOfEvent * iEnd);
				GpuEvent * pDestinationEvent = (GpuEvent*) pDestinationEventBuffer;

				if(pDestinationEvent->i_Type != GpuEvent::NONE) // there is an event in destination position
				{
					iPrevToEnd = iEnd;
					iEnd -= iExitEventCount;
				}
				else
				{
					break;
				}

			}

			iEnd = (iEnd < 0 ? iPrevToEnd : iEnd);

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

// ===============================================================================================================================

GpuLengthSlidingWindowFirstKernel::GpuLengthSlidingWindowFirstKernel(GpuProcessor * _pProc, GpuProcessorContext * _pContext,
		int _iThreadBlockSize, int _iWindowSize, FILE * _fPLog) :
	GpuKernel(_pProc, _pContext->GetDeviceId(), _iThreadBlockSize, _fPLog),
	p_Context(_pContext),
	p_InputEventBuffer(NULL),
	p_ResultEventBuffer(NULL),
	p_WindowEventBuffer(NULL),
	b_DeviceSet(false),
	i_WindowSize(_iWindowSize),
	i_RemainingCount(_iWindowSize)
{

}

GpuLengthSlidingWindowFirstKernel::~GpuLengthSlidingWindowFirstKernel()
{
	fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] destroy\n");
	fflush(fp_Log);

	if(p_DeviceOutputAttributeMapping)
	{
		CUDA_CHECK_RETURN(cudaFree(p_DeviceOutputAttributeMapping));
		p_DeviceOutputAttributeMapping = NULL;
	}
}

bool GpuLengthSlidingWindowFirstKernel::Initialize(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize)
{
	fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] Initialize : StreamIndex=%d \n", _iStreamIndex);
	fflush(fp_Log);

	// set input event buffer
	fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] InpuEventBufferIndex=%d\n", i_InputBufferIndex);
	fflush(fp_Log);
	p_InputEventBuffer = (GpuStreamEventBuffer*) p_Context->GetEventBuffer(i_InputBufferIndex);
	p_InputEventBuffer->Print();

	// set resulting event buffer and its meta data
	p_ResultEventBuffer = new GpuStreamEventBuffer("WindowResultEventBuffer", p_Context->GetDeviceId(), _pMetaEvent, fp_Log);
	p_ResultEventBuffer->CreateEventBuffer(_iInputEventBufferSize * 2);

	i_ResultEventBufferIndex = p_Context->AddEventBuffer(p_ResultEventBuffer);

	fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] ResultEventBuffer created : Index=%d Size=%d bytes\n", i_ResultEventBufferIndex,
			p_ResultEventBuffer->GetEventBufferSizeInBytes());
	fflush(fp_Log);
	p_ResultEventBuffer->Print();

	p_WindowEventBuffer = new GpuStreamEventBuffer("WindowEventBuffer", p_Context->GetDeviceId(), _pMetaEvent, fp_Log);
	p_WindowEventBuffer->CreateEventBuffer(i_WindowSize);

	fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] Created device window buffer : Length=%d Size=%d bytes\n", i_WindowSize,
			p_WindowEventBuffer->GetEventBufferSizeInBytes());
	fflush(fp_Log);

	fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] initialize window buffer data \n");
	fflush(fp_Log);
	p_WindowEventBuffer->Print();

	p_WindowEventBuffer->ResetHostEventBuffer(0);

	char * pHostWindowBuffer = p_WindowEventBuffer->GetHostEventBuffer();
	char * pCurrentEvent;
	for(int i=0; i<i_WindowSize; ++i)
	{
		pCurrentEvent = pHostWindowBuffer + (_pMetaEvent->i_SizeOfEventInBytes * i);
		GpuEvent * pGpuEvent = (GpuEvent*) pCurrentEvent;
		pGpuEvent->i_Type = GpuEvent::NONE;
	}

	p_WindowEventBuffer->CopyToDevice(false);

	// copy Output mappings
	if(p_HostOutputAttributeMapping)
	{
		fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] Copying AttributeMappings to device \n");
		fflush(fp_Log);

		fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] AttributeMapCount : %d \n", p_HostOutputAttributeMapping->i_MappingCount);
		for(int c=0; c<p_HostOutputAttributeMapping->i_MappingCount; ++c)
		{
			fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] Map : Form [Stream=%d, Attrib=%d] To [Attrib=%d] \n",
					p_HostOutputAttributeMapping->p_Mappings[c].from[AttributeMapping::STREAM_INDEX],
					p_HostOutputAttributeMapping->p_Mappings[c].from[AttributeMapping::ATTRIBUTE_INDEX],
					p_HostOutputAttributeMapping->p_Mappings[c].to);

		}

		CUDA_CHECK_RETURN(cudaMalloc(
				(void**) &p_DeviceOutputAttributeMapping,
				sizeof(AttributeMappings)));

		AttributeMappings * pHostMappings = (AttributeMappings*) malloc(sizeof(AttributeMappings));
		pHostMappings->i_MappingCount = p_HostOutputAttributeMapping->i_MappingCount;
		pHostMappings->p_Mappings = NULL;

		CUDA_CHECK_RETURN(cudaMalloc(
				(void**) &pHostMappings->p_Mappings,
				sizeof(AttributeMapping) * p_HostOutputAttributeMapping->i_MappingCount));

		CUDA_CHECK_RETURN(cudaMemcpy(
				pHostMappings->p_Mappings,
				p_HostOutputAttributeMapping->p_Mappings,
				sizeof(AttributeMapping) * p_HostOutputAttributeMapping->i_MappingCount,
				cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(
				p_DeviceOutputAttributeMapping,
				pHostMappings,
				sizeof(AttributeMappings),
				cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		CUDA_CHECK_RETURN(cudaThreadSynchronize());

		free(pHostMappings);
		pHostMappings = NULL;
	}

	fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] Initialization complete\n");
	fflush(fp_Log);

	return true;
}

void GpuLengthSlidingWindowFirstKernel::Process(int _iStreamIndex, int & _iNumEvents)
{
#if GPU_DEBUG >= GPU_DEBUG_LEVEL_TRACE
	fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] Process : EventCount=%d WindowRemainingCount=%d\n", _iNumEvents, i_RemainingCount);
	fflush(fp_Log);

	GpuUtils::PrintByteBuffer(p_InputEventBuffer->GetHostEventBuffer(), _iNumEvents, p_InputEventBuffer->GetHostMetaEvent(),
			"GpuLengthSlidingWindowFirstKernel::In", fp_Log);
#endif

	if(!b_DeviceSet)
	{
		GpuCudaHelper::SelectDevice(i_DeviceId, "GpuLengthSlidingWindowFirstKernel", fp_Log);
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

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] Invoke kernel Blocks(%d,%d) Threads(%d,%d)\n", numBlocksX, numBlocksY, i_ThreadBlockSize, 1);
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

	if(b_LastKernel)
	{
		p_ResultEventBuffer->CopyToHost(true);
#if GPU_DEBUG >= GPU_DEBUG_LEVEL_DEBUG
	fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] Results copied \n");
	fflush(fp_Log);
#endif
	}

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

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_TRACE
	if(b_LastKernel)
	{
		GpuUtils::PrintByteBuffer(p_ResultEventBuffer->GetHostEventBuffer(), _iNumEvents * 2, p_ResultEventBuffer->GetHostMetaEvent(),
				"GpuLengthSlidingWindowFirstKernel::Out", fp_Log);
	}
	fflush(fp_Log);
#endif

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] Kernel complete \n");
	fflush(fp_Log);
#endif


#ifdef KERNEL_TIME
	sdkStopTimer(&p_StopWatch);
	float fElapsed = sdkGetTimerValue(&p_StopWatch);
	fprintf(fp_Log, "[GpuLengthSlidingWindowFirstKernel] Stats : Elapsed=%f ms\n", fElapsed);
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
}

char * GpuLengthSlidingWindowFirstKernel::GetResultEventBuffer()
{
	return p_ResultEventBuffer->GetHostEventBuffer();
}

int GpuLengthSlidingWindowFirstKernel::GetResultEventBufferSize()
{
	return p_ResultEventBuffer->GetEventBufferSizeInBytes();
}

// ===============================================================================================================================

GpuLengthSlidingWindowFilterKernel::GpuLengthSlidingWindowFilterKernel(GpuProcessor * _pProc, GpuProcessorContext * _pContext,
		int _iThreadBlockSize, int _iWindowSize, FILE * _fPLog) :
	GpuKernel(_pProc, _pContext->GetDeviceId(), _iThreadBlockSize, _fPLog),
	p_Context(_pContext),
	p_InputEventBuffer(NULL),
	p_ResultEventBuffer(NULL),
	p_WindowEventBuffer(NULL),
	b_DeviceSet(false),
	i_WindowSize(_iWindowSize),
	i_RemainingCount(_iWindowSize)
{

}

GpuLengthSlidingWindowFilterKernel::~GpuLengthSlidingWindowFilterKernel()
{
	fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] destroy\n");
	fflush(fp_Log);

	if(p_DeviceOutputAttributeMapping)
	{
		CUDA_CHECK_RETURN(cudaFree(p_DeviceOutputAttributeMapping));
		p_DeviceOutputAttributeMapping = NULL;
	}
}

bool GpuLengthSlidingWindowFilterKernel::Initialize(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize)
{
	fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] Initialize : StreamIndex=%d \n", _iStreamIndex);
	fflush(fp_Log);

	// set input event buffer
	fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] InpuEventBufferIndex=%d\n", i_InputBufferIndex);
	fflush(fp_Log);
	p_InputEventBuffer = (GpuStreamEventBuffer*) p_Context->GetEventBuffer(i_InputBufferIndex);
	p_InputEventBuffer->Print();

	// set resulting event buffer and its meta data
	p_ResultEventBuffer = new GpuStreamEventBuffer("WindowResultEventBuffer", p_Context->GetDeviceId(), _pMetaEvent, fp_Log);
	p_ResultEventBuffer->CreateEventBuffer(_iInputEventBufferSize * 2);

	i_ResultEventBufferIndex = p_Context->AddEventBuffer(p_ResultEventBuffer);

	fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] ResultEventBuffer created : Index=%d Size=%d bytes\n", i_ResultEventBufferIndex,
			p_ResultEventBuffer->GetEventBufferSizeInBytes());
	fflush(fp_Log);
	p_ResultEventBuffer->Print();

	p_WindowEventBuffer = new GpuStreamEventBuffer("WindowEventBuffer", p_Context->GetDeviceId(), _pMetaEvent, fp_Log);
	p_WindowEventBuffer->CreateEventBuffer(i_WindowSize);

	fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] Created device window buffer : Length=%d Size=%d bytes\n", i_WindowSize,
			p_WindowEventBuffer->GetEventBufferSizeInBytes());
	fflush(fp_Log);
	p_WindowEventBuffer->Print();

	p_WindowEventBuffer->ResetHostEventBuffer(0);

	char * pHostWindowBuffer = p_WindowEventBuffer->GetHostEventBuffer();
	char * pCurrentEvent;
	for(int i=0; i<i_WindowSize; ++i)
	{
		pCurrentEvent = pHostWindowBuffer + (_pMetaEvent->i_SizeOfEventInBytes * i);
		GpuEvent * pGpuEvent = (GpuEvent*) pCurrentEvent;
		pGpuEvent->i_Type = GpuEvent::NONE;
	}

	p_WindowEventBuffer->CopyToDevice(false);

	// copy Output mappings
	if(p_HostOutputAttributeMapping)
	{
		fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] Copying AttributeMappings to device \n");
		fflush(fp_Log);

		fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] AttributeMapCount : %d \n", p_HostOutputAttributeMapping->i_MappingCount);
		for(int c=0; c<p_HostOutputAttributeMapping->i_MappingCount; ++c)
		{
			fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] Map : Form [Stream=%d, Attrib=%d] To [Attrib=%d] \n",
					p_HostOutputAttributeMapping->p_Mappings[c].from[AttributeMapping::STREAM_INDEX],
					p_HostOutputAttributeMapping->p_Mappings[c].from[AttributeMapping::ATTRIBUTE_INDEX],
					p_HostOutputAttributeMapping->p_Mappings[c].to);

		}

		CUDA_CHECK_RETURN(cudaMalloc(
				(void**) &p_DeviceOutputAttributeMapping,
				sizeof(AttributeMappings)));

		AttributeMappings * pHostMappings = (AttributeMappings*) malloc(sizeof(AttributeMappings));
		pHostMappings->i_MappingCount = p_HostOutputAttributeMapping->i_MappingCount;
		pHostMappings->p_Mappings = NULL;

		CUDA_CHECK_RETURN(cudaMalloc(
				(void**) &pHostMappings->p_Mappings,
				sizeof(AttributeMapping) * p_HostOutputAttributeMapping->i_MappingCount));

		CUDA_CHECK_RETURN(cudaMemcpy(
				pHostMappings->p_Mappings,
				p_HostOutputAttributeMapping->p_Mappings,
				sizeof(AttributeMapping) * p_HostOutputAttributeMapping->i_MappingCount,
				cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(
				p_DeviceOutputAttributeMapping,
				pHostMappings,
				sizeof(AttributeMappings),
				cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		CUDA_CHECK_RETURN(cudaThreadSynchronize());

		free(pHostMappings);
		pHostMappings = NULL;
	}

	fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] Initialization complete\n");
	fflush(fp_Log);

	return true;
}

void GpuLengthSlidingWindowFilterKernel::Process(int _iStreamIndex, int & _iNumEvents)
{
#if GPU_DEBUG >= GPU_DEBUG_LEVEL_TRACE
	fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] Process : EventCount=%d\n", _iNumEvents);
	fflush(fp_Log);

	p_InputEventBuffer->CopyToHost(true);
	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	GpuUtils::PrintByteBuffer(p_InputEventBuffer->GetHostEventBuffer(), _iNumEvents, p_InputEventBuffer->GetHostMetaEvent(),
			"GpuLengthSlidingWindowFilterKernel::In", fp_Log);
#endif

	if(!b_DeviceSet)
	{
		GpuCudaHelper::SelectDevice(i_DeviceId, "GpuLengthSlidingWindowFilterKernel", fp_Log);
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

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] Invoke kernel Blocks(%d,%d) Threads(%d,%d)\n", numBlocksX, numBlocksY, i_ThreadBlockSize, 1);
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

	if(b_LastKernel)
	{
		p_ResultEventBuffer->CopyToHost(true);

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_DEBUG
	fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] Results copied \n");
	fflush(fp_Log);
#endif
	}

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

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_TRACE
	if(b_LastKernel)
	{
		GpuUtils::PrintByteBuffer(p_ResultEventBuffer->GetHostEventBuffer(), _iNumEvents * 2, p_ResultEventBuffer->GetHostMetaEvent(),
				"GpuLengthSlidingWindowFilterKernel::Out", fp_Log);
	}
	fflush(fp_Log);
#endif

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] Kernel complete \n");
	fflush(fp_Log);
#endif


#ifdef KERNEL_TIME
	sdkStopTimer(&p_StopWatch);
	float fElapsed = sdkGetTimerValue(&p_StopWatch);
	fprintf(fp_Log, "[GpuLengthSlidingWindowFilterKernel] Stats : Elapsed=%f ms\n", fElapsed);
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
}

char * GpuLengthSlidingWindowFilterKernel::GetResultEventBuffer()
{
	return p_ResultEventBuffer->GetHostEventBuffer();
}

int GpuLengthSlidingWindowFilterKernel::GetResultEventBufferSize()
{
	return p_ResultEventBuffer->GetEventBufferSizeInBytes();
}

}

#endif
