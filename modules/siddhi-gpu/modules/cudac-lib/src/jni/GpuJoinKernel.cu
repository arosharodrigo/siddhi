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
#include "GpuJoinKernelCore.h"
#include "GpuFilterProcessor.h"
#include "GpuUtils.h"

namespace SiddhiGpu
{

// process batch of events in one stream of join processor
__global__
void ProcessEventsJoinLeftTriggerAllOn(
		char               * _pInputEventBuffer,         // input events buffer
		GpuKernelMetaEvent * _pInputMetaEvent,           // Meta event for input events
		int                  _iInputNumberOfEvents,      // Number of events in input buffer
		char               * _pEventWindowBuffer,        // Event window buffer of this stream
		int                  _iWindowLength,             // Length of current events window
		int                  _iRemainingCount,           // Remaining free slots in Window buffer
		GpuKernelMetaEvent * _pOtherStreamMetaEvent,     // Meta event for other stream
		char               * _pOtherEventWindowBuffer,   // Event window buffer of other stream
		int                  _iOtherWindowLength,        // Length of current events window of other stream
		int                  _iOtherRemainingCount,      // Remaining free slots in Window buffer of other stream
		GpuKernelFilter    * _pOnCompareFilter,          // OnCompare filter buffer - pre-copied at initialization
		uint64_t             _iWithInTime,               // WithIn time in milliseconds
		GpuKernelMetaEvent * _pOutputStreamMetaEvent,    // Meta event for output stream
		char               * _pResultsBuffer,            // Resulting events buffer for this stream
		AttributeMappings  * _pOutputAttribMappings,     // Output event attribute mappings
		int                  _iEventsPerBlock            // number of events allocated per block
)
{
	// avoid out of bound threads
	if(threadIdx.x >= _iEventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _iInputNumberOfEvents / _iEventsPerBlock) && // last thread block
			(threadIdx.x >= _iInputNumberOfEvents % _iEventsPerBlock)) // extra threads
	{
		return;
	}

	// get assigned event
	int iEventIdx = (blockIdx.x * _iEventsPerBlock) + threadIdx.x;

	// get in event starting position
	char * pInEventBuffer = _pInputEventBuffer + (_pInputMetaEvent->i_SizeOfEventInBytes * iEventIdx);

	// output to results buffer [in event, expired event]
	// {other stream event size * other window size} * 2 (for in/exp)
	int iOutputSegmentSize = _pOutputStreamMetaEvent->i_SizeOfEventInBytes * _iOtherWindowLength * 2;

	char * pResultsInEventBufferSegment = _pResultsBuffer + (iOutputSegmentSize * iEventIdx);
	char * pResultsExpiredEventBufferSegment = pResultsInEventBufferSegment + (iOutputSegmentSize / 2);

	char * pExpiredEventBuffer = NULL;
	GpuEvent * pExpiredEvent = NULL;

	GpuEvent * pInEvent = (GpuEvent*) pInEventBuffer;

	// calculate in/expired event pair for this event

	if(iEventIdx >= _iRemainingCount)
	{
		if(iEventIdx < _iWindowLength)
		{
			// in window buffer
			char * pExpiredOutEventInWindowBuffer = _pEventWindowBuffer + (_pInputMetaEvent->i_SizeOfEventInBytes * (iEventIdx - _iRemainingCount));

			GpuEvent * pWindowEvent = (GpuEvent*) pExpiredOutEventInWindowBuffer;
			if(pWindowEvent->i_Type != GpuEvent::NONE) // if window event is filled
			{
				pExpiredEventBuffer = pExpiredOutEventInWindowBuffer;

			}
			else
			{
				// no expiring event
			}
		}
		else
		{
			// in input event buffer
			char * pExpiredOutEventInInputBuffer = _pInputEventBuffer + (_pInputMetaEvent->i_SizeOfEventInBytes * (iEventIdx - _iWindowLength));
			pExpiredEventBuffer = pExpiredOutEventInInputBuffer;
		}
	}
	else
	{
		// [NULL,inEvent]
		// no expiring event
	}


	// get all matching event for in event from other window buffer and copy them to output event buffer

	// for each events in other window
	int iOtherWindowFillCount  = _iOtherWindowLength - _iOtherRemainingCount;
	int iMatchedCount = 0;
	for(int i=0; i<iOtherWindowFillCount; ++i)
	{
		// get other window event
		char * pOtherWindowEventBuffer = _pOtherEventWindowBuffer + (_pOtherStreamMetaEvent->i_SizeOfEventInBytes * i);
		GpuEvent * pOtherWindowEvent = (GpuEvent*) pOtherWindowEventBuffer;

		// get buffer position for in event matching results
		char * pResultInMatchingEventBuffer = pResultsInEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
		GpuEvent * pResultInMatchingEvent = (GpuEvent*) pResultInMatchingEventBuffer;

		if(pInEvent->i_Sequence > pOtherWindowEvent->i_Sequence &&
				(pInEvent->i_Timestamp - pOtherWindowEvent->i_Timestamp) <= _iWithInTime)
		{
			ExpressionEvalParameters mExpressionParam;
			mExpressionParam.p_OnCompare = _pOnCompareFilter;
			mExpressionParam.a_Meta[0] = _pInputMetaEvent;
			mExpressionParam.a_Event[0] = pInEventBuffer;
			mExpressionParam.a_Meta[1] = _pOtherStreamMetaEvent;
			mExpressionParam.a_Event[1] = pOtherWindowEventBuffer;
			mExpressionParam.i_CurrentIndex = 0;

			bool bOnCompareMatched = Evaluate(mExpressionParam);
			if(bOnCompareMatched)
			{
				// copy output event to buffer - map attributes from input streams to output stream
				pResultInMatchingEvent->i_Type = GpuEvent::CURRENT;
				pResultInMatchingEvent->i_Sequence = pInEvent->i_Sequence;
				pResultInMatchingEvent->i_Timestamp = pInEvent->i_Timestamp;

				for(int m=0; m < _pOutputAttribMappings->i_MappingCount; ++m)
				{
					int iFromStreamIndex = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::STREAM_INDEX];
					int iFromAttrib = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::ATTRIBUTE_INDEX];
					int iTo = _pOutputAttribMappings->p_Mappings[m].to;

					memcpy(
						pResultInMatchingEventBuffer + _pOutputStreamMetaEvent->p_Attributes[iTo].i_Position, // to
						mExpressionParam.a_Event[iFromStreamIndex] + mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Position, // from
						mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Length // size
					);
				}

				iMatchedCount++;
			}
		}
		else
		{
			// cannot continue, last result event for this segment
			pResultInMatchingEvent->i_Type = GpuEvent::RESET;
			break;
		}
	}

	if(iMatchedCount < iOtherWindowFillCount || iOtherWindowFillCount == 0)
	{
		char * pResultInMatchingEventBuffer = pResultsInEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
		GpuEvent * pResultInMatchingEvent = (GpuEvent*) pResultInMatchingEventBuffer;
		pResultInMatchingEvent->i_Type = GpuEvent::RESET;
	}

	if(pExpiredEventBuffer != NULL)
	{
		pExpiredEvent = (GpuEvent*) pExpiredEventBuffer;

		iMatchedCount = 0;
		// for each events in other window
		for(int i=0; i<iOtherWindowFillCount; ++i)
		{
			// get other window event
			char * pOtherWindowEventBuffer = _pOtherEventWindowBuffer + (_pOtherStreamMetaEvent->i_SizeOfEventInBytes * i);
			GpuEvent * pOtherWindowEvent = (GpuEvent*) pOtherWindowEventBuffer;

			// get buffer position for expire event matching results
			char * pResultExpireMatchingEventBuffer = pResultsExpiredEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
			GpuEvent * pResultExpireMatchingEvent = (GpuEvent*) pResultExpireMatchingEventBuffer;

			if(pExpiredEvent->i_Sequence < pOtherWindowEvent->i_Sequence &&
					(pOtherWindowEvent->i_Timestamp - pExpiredEvent->i_Timestamp) <= _iWithInTime)
			{
				ExpressionEvalParameters mExpressionParam;
				mExpressionParam.p_OnCompare = _pOnCompareFilter;
				mExpressionParam.a_Meta[0] = _pInputMetaEvent;
				mExpressionParam.a_Event[0] = pExpiredEventBuffer;
				mExpressionParam.a_Meta[1] = _pOtherStreamMetaEvent;
				mExpressionParam.a_Event[1] = pOtherWindowEventBuffer;
				mExpressionParam.i_CurrentIndex = 0;

				bool bOnCompareMatched = Evaluate(mExpressionParam);
				if(bOnCompareMatched)
				{
					// copy output event to buffer - map attributes from input streams to output stream
					pResultExpireMatchingEvent->i_Type = GpuEvent::EXPIRED;
					pResultExpireMatchingEvent->i_Sequence = pExpiredEvent->i_Sequence;
					pResultExpireMatchingEvent->i_Timestamp = pExpiredEvent->i_Timestamp;

					for(int m=0; m < _pOutputAttribMappings->i_MappingCount; ++m)
					{
						int iFromStreamIndex = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::STREAM_INDEX];
						int iFromAttrib = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::ATTRIBUTE_INDEX];
						int iTo = _pOutputAttribMappings->p_Mappings[m].to;

						memcpy(
								pResultExpireMatchingEventBuffer + _pOutputStreamMetaEvent->p_Attributes[iTo].i_Position, // to
								mExpressionParam.a_Event[iFromStreamIndex] + mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Position, // from
								mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Length // size
						);
					}

					iMatchedCount++;
				}
			}
			else
			{
				// cannot continue, last result event for this segment
				pResultExpireMatchingEvent->i_Type = GpuEvent::RESET;
				break;
			}
		}

		if(iMatchedCount < iOtherWindowFillCount || iOtherWindowFillCount == 0)
		{
			char * pResultExpireMatchingEventBuffer = pResultsExpiredEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
			GpuEvent * pResultExpireMatchingEvent = (GpuEvent*) pResultExpireMatchingEventBuffer;
			pResultExpireMatchingEvent->i_Type = GpuEvent::RESET;
		}
	}

}

__global__
void ProcessEventsJoinLeftTriggerCurrentOn(
		char               * _pInputEventBuffer,         // input events buffer
		GpuKernelMetaEvent * _pInputMetaEvent,           // Meta event for input events
		int                  _iInputNumberOfEvents,      // Number of events in input buffer
		char               * _pEventWindowBuffer,        // Event window buffer of this stream
		int                  _iWindowLength,             // Length of current events window
		int                  _iRemainingCount,           // Remaining free slots in Window buffer
		GpuKernelMetaEvent * _pOtherStreamMetaEvent,     // Meta event for other stream
		char               * _pOtherEventWindowBuffer,   // Event window buffer of other stream
		int                  _iOtherWindowLength,        // Length of current events window of other stream
		int                  _iOtherRemainingCount,      // Remaining free slots in Window buffer of other stream
		GpuKernelFilter    * _pOnCompareFilter,          // OnCompare filter buffer - pre-copied at initialization
		uint64_t             _iWithInTime,               // WithIn time in milliseconds
		GpuKernelMetaEvent * _pOutputStreamMetaEvent,    // Meta event for output stream
		char               * _pResultsBuffer,            // Resulting events buffer for this stream
		AttributeMappings  * _pOutputAttribMappings,     // Output event attribute mappings
		int                  _iEventsPerBlock,           // number of events allocated per block
		int                  _iWorkSize                  // Number of events in window process by this kernel
)
{
	// avoid out of bound threads
	if(threadIdx.x >= _iEventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _iInputNumberOfEvents / _iEventsPerBlock) && // last thread block
			(threadIdx.x >= _iInputNumberOfEvents % _iEventsPerBlock)) // extra threads
	{
		return;
	}

	// get assigned event
	int iGlobalThreadIdx = (blockIdx.x * _iEventsPerBlock) + threadIdx.x;

	// get in buffer index
	int iWorkerCount = ceil((float)_iOtherWindowLength / _iWorkSize);
	int iInEventIndex = iGlobalThreadIdx / iWorkerCount;
	int iWindowStartEventIndex = (iGlobalThreadIdx % iWorkerCount) * _iWorkSize;

	// get in event starting position
	char * pInEventBuffer = _pInputEventBuffer + (_pInputMetaEvent->i_SizeOfEventInBytes * iInEventIndex);

	// output to results buffer [in event, expired event]
	// {other stream event size * other window size} * 2 (for in/exp)
//	int iOutputSegmentSize = _pOutputStreamMetaEvent->i_SizeOfEventInBytes * _iOtherWindowLength;

	char * pResultsInEventBufferSegment = _pResultsBuffer + (iGlobalThreadIdx * _iWorkSize * _pOutputStreamMetaEvent->i_SizeOfEventInBytes);
//			+ (iWindowStartEventIndex * _pOutputStreamMetaEvent->i_SizeOfEventInBytes);

	GpuEvent * pInEvent = (GpuEvent*) pInEventBuffer;

	memset(pResultsInEventBufferSegment, 0, _iWorkSize * _pOutputStreamMetaEvent->i_SizeOfEventInBytes);

	// get all matching event for in event from other window buffer and copy them to output event buffer

	// for each events in other window
	int iOtherWindowFillCount  = _iOtherWindowLength - _iOtherRemainingCount;

	if(iWindowStartEventIndex < iOtherWindowFillCount)
	{
		int iWindowEndEventIndex = min(iWindowStartEventIndex + _iWorkSize, iOtherWindowFillCount);

		int iMatchedCount = 0;
		for(int i=iWindowStartEventIndex; i<iWindowEndEventIndex; ++i)
		{
			// get other window event
			char * pOtherWindowEventBuffer = _pOtherEventWindowBuffer + (_pOtherStreamMetaEvent->i_SizeOfEventInBytes * i);
			GpuEvent * pOtherWindowEvent = (GpuEvent*) pOtherWindowEventBuffer;

			// get buffer position for in event matching results
			char * pResultInMatchingEventBuffer = pResultsInEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
			GpuEvent * pResultInMatchingEvent = (GpuEvent*) pResultInMatchingEventBuffer;

			if(pInEvent->i_Sequence > pOtherWindowEvent->i_Sequence &&
					(pInEvent->i_Timestamp - pOtherWindowEvent->i_Timestamp) <= _iWithInTime)
			{
				ExpressionEvalParameters mExpressionParam;
				mExpressionParam.p_OnCompare = _pOnCompareFilter;
				mExpressionParam.a_Meta[0] = _pInputMetaEvent;
				mExpressionParam.a_Event[0] = pInEventBuffer;
				mExpressionParam.a_Meta[1] = _pOtherStreamMetaEvent;
				mExpressionParam.a_Event[1] = pOtherWindowEventBuffer;
				mExpressionParam.i_CurrentIndex = 0;

				bool bOnCompareMatched = Evaluate(mExpressionParam);
				if(bOnCompareMatched)
				{
					// copy output event to buffer - map attributes from input streams to output stream
					//

					pResultInMatchingEvent->i_Type = GpuEvent::CURRENT;
					pResultInMatchingEvent->i_Sequence = pInEvent->i_Sequence;
					pResultInMatchingEvent->i_Timestamp = pInEvent->i_Timestamp;

					//TODO: #pragma __unroll__ 5
					for(int m=0; m < _pOutputAttribMappings->i_MappingCount; ++m)
					{
						int iFromStreamIndex = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::STREAM_INDEX];
						int iFromAttrib = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::ATTRIBUTE_INDEX];
						int iTo = _pOutputAttribMappings->p_Mappings[m].to;

						memcpy(
								pResultInMatchingEventBuffer + _pOutputStreamMetaEvent->p_Attributes[iTo].i_Position, // to
								mExpressionParam.a_Event[iFromStreamIndex] + mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Position, // from
								mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Length // size
						);
					}

					iMatchedCount++;
				}
			}
			else
			{
				// cannot continue, last result event for this segment
				pResultInMatchingEvent->i_Type = GpuEvent::RESET;
				break;
			}
		}

		if(iMatchedCount < (iWindowEndEventIndex - iWindowStartEventIndex))
		{
			char * pResultInMatchingEventBuffer = pResultsInEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
			GpuEvent * pResultInMatchingEvent = (GpuEvent*) pResultInMatchingEventBuffer;
			pResultInMatchingEvent->i_Type = GpuEvent::RESET;
		}

	}
	else
	{
		GpuEvent * pResultInMatchingEvent = (GpuEvent*) pResultsInEventBufferSegment;
		pResultInMatchingEvent->i_Type = GpuEvent::RESET;
	}
}

__global__
void ProcessEventsJoinLeftTriggerExpiredOn(
		char               * _pInputEventBuffer,         // input events buffer
		GpuKernelMetaEvent * _pInputMetaEvent,           // Meta event for input events
		int                  _iInputNumberOfEvents,      // Number of events in input buffer
		char               * _pEventWindowBuffer,        // Event window buffer of this stream
		int                  _iWindowLength,             // Length of current events window
		int                  _iRemainingCount,           // Remaining free slots in Window buffer
		GpuKernelMetaEvent * _pOtherStreamMetaEvent,     // Meta event for other stream
		char               * _pOtherEventWindowBuffer,   // Event window buffer of other stream
		int                  _iOtherWindowLength,        // Length of current events window of other stream
		int                  _iOtherRemainingCount,      // Remaining free slots in Window buffer of other stream
		GpuKernelFilter    * _pOnCompareFilter,          // OnCompare filter buffer - pre-copied at initialization
		uint64_t             _iWithInTime,               // WithIn time in milliseconds
		GpuKernelMetaEvent * _pOutputStreamMetaEvent,    // Meta event for output stream
		char               * _pResultsBuffer,            // Resulting events buffer for this stream
		AttributeMappings  * _pOutputAttribMappings,     // Output event attribute mappings
		int                  _iEventsPerBlock            // number of events allocated per block
)
{
	// avoid out of bound threads
	if(threadIdx.x >= _iEventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _iInputNumberOfEvents / _iEventsPerBlock) && // last thread block
			(threadIdx.x >= _iInputNumberOfEvents % _iEventsPerBlock)) // extra threads
	{
		return;
	}

	// get assigned event
	int iEventIdx = (blockIdx.x * _iEventsPerBlock) + threadIdx.x;

	// output to results buffer [in event, expired event]
	// {other stream event size * other window size} * 2 (for in/exp)
	int iOutputSegmentSize = _pOutputStreamMetaEvent->i_SizeOfEventInBytes * _iOtherWindowLength;

	char * pResultsExpiredEventBufferSegment = _pResultsBuffer + (iOutputSegmentSize * iEventIdx);

	char * pExpiredEventBuffer = NULL;
	GpuEvent * pExpiredEvent = NULL;

	// calculate in/expired event pair for this event

	if(iEventIdx >= _iRemainingCount)
	{
		if(iEventIdx < _iWindowLength)
		{
			// in window buffer
			char * pExpiredOutEventInWindowBuffer = _pEventWindowBuffer + (_pInputMetaEvent->i_SizeOfEventInBytes * (iEventIdx - _iRemainingCount));

			GpuEvent * pWindowEvent = (GpuEvent*) pExpiredOutEventInWindowBuffer;
			if(pWindowEvent->i_Type != GpuEvent::NONE) // if window event is filled
			{
				pExpiredEventBuffer = pExpiredOutEventInWindowBuffer;

			}
			else
			{
				// no expiring event
			}
		}
		else
		{
			// in input event buffer
			char * pExpiredOutEventInInputBuffer = _pInputEventBuffer + (_pInputMetaEvent->i_SizeOfEventInBytes * (iEventIdx - _iWindowLength));
			pExpiredEventBuffer = pExpiredOutEventInInputBuffer;
		}
	}
	else
	{
		// [NULL,inEvent]
		// no expiring event
	}

	if(pExpiredEventBuffer != NULL)
	{
		pExpiredEvent = (GpuEvent*) pExpiredEventBuffer;

		// for each events in other window
		//	 get all matching event for in event from other window buffer and copy them to output event buffer
		int iOtherWindowFillCount  = _iOtherWindowLength - _iOtherRemainingCount;
		int iMatchedCount = 0;
		// for each events in other window
		for(int i=0; i<iOtherWindowFillCount; ++i)
		{
			// get other window event
			char * pOtherWindowEventBuffer = _pOtherEventWindowBuffer + (_pOtherStreamMetaEvent->i_SizeOfEventInBytes * i);
			GpuEvent * pOtherWindowEvent = (GpuEvent*) pOtherWindowEventBuffer;

			// get buffer position for expire event matching results
			char * pResultExpireMatchingEventBuffer = pResultsExpiredEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
			GpuEvent * pResultExpireMatchingEvent = (GpuEvent*) pResultExpireMatchingEventBuffer;

			if(pExpiredEvent->i_Sequence < pOtherWindowEvent->i_Sequence &&
					(pOtherWindowEvent->i_Timestamp - pExpiredEvent->i_Timestamp) <= _iWithInTime)
			{
				ExpressionEvalParameters mExpressionParam;
				mExpressionParam.p_OnCompare = _pOnCompareFilter;
				mExpressionParam.a_Meta[0] = _pInputMetaEvent;
				mExpressionParam.a_Event[0] = pExpiredEventBuffer;
				mExpressionParam.a_Meta[1] = _pOtherStreamMetaEvent;
				mExpressionParam.a_Event[1] = pOtherWindowEventBuffer;
				mExpressionParam.i_CurrentIndex = 0;

				bool bOnCompareMatched = Evaluate(mExpressionParam);
				if(bOnCompareMatched)
				{
					// copy output event to buffer - map attributes from input streams to output stream
					pResultExpireMatchingEvent->i_Type = GpuEvent::EXPIRED;
					pResultExpireMatchingEvent->i_Sequence = pExpiredEvent->i_Sequence;
					pResultExpireMatchingEvent->i_Timestamp = pExpiredEvent->i_Timestamp;

					for(int m=0; m < _pOutputAttribMappings->i_MappingCount; ++m)
					{
						int iFromStreamIndex = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::STREAM_INDEX];
						int iFromAttrib = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::ATTRIBUTE_INDEX];
						int iTo = _pOutputAttribMappings->p_Mappings[m].to;

						memcpy(
								pResultExpireMatchingEventBuffer + _pOutputStreamMetaEvent->p_Attributes[iTo].i_Position, // to
								mExpressionParam.a_Event[iFromStreamIndex] + mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Position, // from
								mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Length // size
						);
					}

					iMatchedCount++;
				}
			}
			else
			{
				// cannot continue, last result event for this segment
				pResultExpireMatchingEvent->i_Type = GpuEvent::RESET;
				break;
			}
		}

		if(iMatchedCount < iOtherWindowFillCount || iOtherWindowFillCount == 0)
		{
			char * pResultExpireMatchingEventBuffer = pResultsExpiredEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
			GpuEvent * pResultExpireMatchingEvent = (GpuEvent*) pResultExpireMatchingEventBuffer;
			pResultExpireMatchingEvent->i_Type = GpuEvent::RESET;
		}
	}

}

__global__
void ProcessEventsJoinRightTriggerAllOn(
		char               * _pInputEventBuffer,         // input events buffer
		GpuKernelMetaEvent * _pInputMetaEvent,           // Meta event for input events
		int                  _iInputNumberOfEvents,      // Number of events in input buffer
		char               * _pEventWindowBuffer,        // Event window buffer of this stream
		int                  _iWindowLength,             // Length of current events window
		int                  _iRemainingCount,           // Remaining free slots in Window buffer
		GpuKernelMetaEvent * _pOtherStreamMetaEvent,     // Meta event for other stream
		char               * _pOtherEventWindowBuffer,   // Event window buffer of other stream
		int                  _iOtherWindowLength,        // Length of current events window of other stream
		int                  _iOtherRemainingCount,      // Remaining free slots in Window buffer of other stream
		GpuKernelFilter    * _pOnCompareFilter,          // OnCompare filter buffer - pre-copied at initialization
		uint64_t             _iWithInTime,               // WithIn time in milliseconds
		GpuKernelMetaEvent * _pOutputStreamMetaEvent,    // Meta event for output stream
		char               * _pResultsBuffer,            // Resulting events buffer for this stream
		AttributeMappings  * _pOutputAttribMappings,     // Output event attribute mappings
		int                  _iEventsPerBlock            // number of events allocated per block
)
{
	// avoid out of bound threads
	if(threadIdx.x >= _iEventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _iInputNumberOfEvents / _iEventsPerBlock) && // last thread block
			(threadIdx.x >= _iInputNumberOfEvents % _iEventsPerBlock)) // extra threads
	{
		return;
	}

	// get assigned event
	int iEventIdx = (blockIdx.x * _iEventsPerBlock) + threadIdx.x;

	// get in event starting position
	char * pInEventBuffer = _pInputEventBuffer + (_pInputMetaEvent->i_SizeOfEventInBytes * iEventIdx);

	// output to results buffer [in event, expired event]
	// {other stream event size * other window size} * 2 (for in/exp)
	int iOutputSegmentSize = _pOutputStreamMetaEvent->i_SizeOfEventInBytes * _iOtherWindowLength * 2;

	char * pResultsInEventBufferSegment = _pResultsBuffer + (iOutputSegmentSize * iEventIdx);
	char * pResultsExpiredEventBufferSegment = pResultsInEventBufferSegment + (iOutputSegmentSize / 2);

	char * pExpiredEventBuffer = NULL;
	GpuEvent * pExpiredEvent = NULL;

	GpuEvent * pInEvent = (GpuEvent*) pInEventBuffer;

	// calculate in/expired event pair for this event

	if(iEventIdx >= _iRemainingCount)
	{
		if(iEventIdx < _iWindowLength)
		{
			// in window buffer
			char * pExpiredOutEventInWindowBuffer = _pEventWindowBuffer + (_pInputMetaEvent->i_SizeOfEventInBytes * (iEventIdx - _iRemainingCount));

			GpuEvent * pWindowEvent = (GpuEvent*) pExpiredOutEventInWindowBuffer;
			if(pWindowEvent->i_Type != GpuEvent::NONE) // if window event is filled
			{
				pExpiredEventBuffer = pExpiredOutEventInWindowBuffer;

			}
			else
			{
				// no expiring event
			}
		}
		else
		{
			// in input event buffer
			char * pExpiredOutEventInInputBuffer = _pInputEventBuffer + (_pInputMetaEvent->i_SizeOfEventInBytes * (iEventIdx - _iWindowLength));
			pExpiredEventBuffer = pExpiredOutEventInInputBuffer;
		}
	}
	else
	{
		// [NULL,inEvent]
		// no expiring event
	}

	// get all matching event for in event from other window buffer and copy them to output event buffer

	// for each events in other window
	int iOtherWindowFillCount  = _iOtherWindowLength - _iOtherRemainingCount;
	int iMatchedCount = 0;
	for(int i=0; i<iOtherWindowFillCount; ++i)
	{
		// get other window event
		char * pOtherWindowEventBuffer = _pOtherEventWindowBuffer + (_pOtherStreamMetaEvent->i_SizeOfEventInBytes * i);
		GpuEvent * pOtherWindowEvent = (GpuEvent*) pOtherWindowEventBuffer;

		// get buffer position for in event matching results
		char * pResultInMatchingEventBuffer = pResultsInEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
		GpuEvent * pResultInMatchingEvent = (GpuEvent*) pResultInMatchingEventBuffer;

		if(pInEvent->i_Sequence > pOtherWindowEvent->i_Sequence &&
				(pInEvent->i_Timestamp - pOtherWindowEvent->i_Timestamp) <= _iWithInTime)
		{
			ExpressionEvalParameters mExpressionParam;
			mExpressionParam.p_OnCompare = _pOnCompareFilter;
			mExpressionParam.a_Meta[0] = _pOtherStreamMetaEvent;
			mExpressionParam.a_Event[0] = pOtherWindowEventBuffer;
			mExpressionParam.a_Meta[1] = _pInputMetaEvent;
			mExpressionParam.a_Event[1] = pInEventBuffer;
			mExpressionParam.i_CurrentIndex = 0;

			bool bOnCompareMatched = Evaluate(mExpressionParam);
			if(bOnCompareMatched)
			{
				// copy output event to buffer - map attributes from input streams to output stream
				pResultInMatchingEvent->i_Type = GpuEvent::CURRENT;
				pResultInMatchingEvent->i_Sequence = pInEvent->i_Sequence;
				pResultInMatchingEvent->i_Timestamp = pInEvent->i_Timestamp;

				for(int m=0; m < _pOutputAttribMappings->i_MappingCount; ++m)
				{
					int iFromStreamIndex = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::STREAM_INDEX];
					int iFromAttrib = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::ATTRIBUTE_INDEX];
					int iTo = _pOutputAttribMappings->p_Mappings[m].to;

					memcpy(
						pResultInMatchingEventBuffer + _pOutputStreamMetaEvent->p_Attributes[iTo].i_Position, // to
						mExpressionParam.a_Event[iFromStreamIndex] + mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Position, // from
						mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Length // size
					);
				}

				iMatchedCount++;
			}
		}
		else
		{
			// cannot continue, last result event for this segment
			pResultInMatchingEvent->i_Type = GpuEvent::RESET;
			break;
		}

		if(iMatchedCount < iOtherWindowFillCount || iOtherWindowFillCount == 0)
		{
			char * pResultInMatchingEventBuffer = pResultsInEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
			GpuEvent * pResultInMatchingEvent = (GpuEvent*) pResultInMatchingEventBuffer;
			pResultInMatchingEvent->i_Type = GpuEvent::RESET;
		}
	}

	if(pExpiredEventBuffer != NULL)
	{
		pExpiredEvent = (GpuEvent*) pExpiredEventBuffer;

		iMatchedCount = 0;
		// for each events in other window
		for(int i=0; i<iOtherWindowFillCount; ++i)
		{
			// get other window event
			char * pOtherWindowEventBuffer = _pOtherEventWindowBuffer + (_pOtherStreamMetaEvent->i_SizeOfEventInBytes * i);
			GpuEvent * pOtherWindowEvent = (GpuEvent*) pOtherWindowEventBuffer;

			// get buffer position for expire event matching results
			char * pResultExpireMatchingEventBuffer = pResultsExpiredEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
			GpuEvent * pResultExpireMatchingEvent = (GpuEvent*) pResultExpireMatchingEventBuffer;

			if(pExpiredEvent->i_Sequence < pOtherWindowEvent->i_Sequence &&
					(pOtherWindowEvent->i_Timestamp - pExpiredEvent->i_Timestamp) <= _iWithInTime)
			{
				ExpressionEvalParameters mExpressionParam;
				mExpressionParam.p_OnCompare = _pOnCompareFilter;
				mExpressionParam.a_Meta[0] = _pOtherStreamMetaEvent;
				mExpressionParam.a_Event[0] = pOtherWindowEventBuffer;
				mExpressionParam.a_Meta[1] = _pInputMetaEvent;
				mExpressionParam.a_Event[1] = pExpiredEventBuffer;
				mExpressionParam.i_CurrentIndex = 0;

				bool bOnCompareMatched = Evaluate(mExpressionParam);
				if(bOnCompareMatched)
				{
					// copy output event to buffer - map attributes from input streams to output stream
					pResultExpireMatchingEvent->i_Type = GpuEvent::EXPIRED;
					pResultExpireMatchingEvent->i_Sequence = pExpiredEvent->i_Sequence;
					pResultExpireMatchingEvent->i_Timestamp = pExpiredEvent->i_Timestamp;

					for(int m=0; m < _pOutputAttribMappings->i_MappingCount; ++m)
					{
						int iFromStreamIndex = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::STREAM_INDEX];
						int iFromAttrib = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::ATTRIBUTE_INDEX];
						int iTo = _pOutputAttribMappings->p_Mappings[m].to;

						memcpy(
								pResultExpireMatchingEventBuffer + _pOutputStreamMetaEvent->p_Attributes[iTo].i_Position, // to
								mExpressionParam.a_Event[iFromStreamIndex] + mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Position, // from
								mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Length // size
						);
					}

					iMatchedCount++;
				}
			}
			else
			{
				// cannot continue, last result event for this segment
				pResultExpireMatchingEvent->i_Type = GpuEvent::RESET;
				break;
			}
		}

		if(iMatchedCount < iOtherWindowFillCount || iOtherWindowFillCount == 0)
		{
			char * pResultExpireMatchingEventBuffer = pResultsExpiredEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
			GpuEvent * pResultExpireMatchingEvent = (GpuEvent*) pResultExpireMatchingEventBuffer;
			pResultExpireMatchingEvent->i_Type = GpuEvent::RESET;
		}
	}

}

__global__
void ProcessEventsJoinRightTriggerCurrentOn(
		char               * _pInputEventBuffer,         // input events buffer
		GpuKernelMetaEvent * _pInputMetaEvent,           // Meta event for input events
		int                  _iInputNumberOfEvents,      // Number of events in input buffer
		char               * _pEventWindowBuffer,        // Event window buffer of this stream
		int                  _iWindowLength,             // Length of current events window
		int                  _iRemainingCount,           // Remaining free slots in Window buffer
		GpuKernelMetaEvent * _pOtherStreamMetaEvent,     // Meta event for other stream
		char               * _pOtherEventWindowBuffer,   // Event window buffer of other stream
		int                  _iOtherWindowLength,        // Length of current events window of other stream
		int                  _iOtherRemainingCount,      // Remaining free slots in Window buffer of other stream
		GpuKernelFilter    * _pOnCompareFilter,          // OnCompare filter buffer - pre-copied at initialization
		uint64_t             _iWithInTime,               // WithIn time in milliseconds
		GpuKernelMetaEvent * _pOutputStreamMetaEvent,    // Meta event for output stream
		char               * _pResultsBuffer,            // Resulting events buffer for this stream
		AttributeMappings  * _pOutputAttribMappings,     // Output event attribute mappings
		int                  _iEventsPerBlock,           // number of events allocated per block
		int                  _iWorkSize                  // Number of events in window process by this kernel
)
{
	// avoid out of bound threads
	if(threadIdx.x >= _iEventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _iInputNumberOfEvents / _iEventsPerBlock) && // last thread block
			(threadIdx.x >= _iInputNumberOfEvents % _iEventsPerBlock)) // extra threads
	{
		return;
	}

	// get assigned event
	int iGlobalThreadIdx = (blockIdx.x * _iEventsPerBlock) + threadIdx.x;

	// get in buffer index
	int iWorkerCount = ceil((float)_iOtherWindowLength / _iWorkSize);
	int iInEventIndex = iGlobalThreadIdx / iWorkerCount;
	int iWindowStartEventIndex = (iGlobalThreadIdx % iWorkerCount) * _iWorkSize;

	// get in event starting position
	char * pInEventBuffer = _pInputEventBuffer + (_pInputMetaEvent->i_SizeOfEventInBytes * iInEventIndex);

	// output to results buffer [in event, expired event]
	// {other stream event size * other window size} * 2 (for in/exp)
//	int iOutputSegmentSizePerEvent = _pOutputStreamMetaEvent->i_SizeOfEventInBytes * _iOtherWindowLength;

	char * pResultsInEventBufferSegment = _pResultsBuffer + (iGlobalThreadIdx * _iWorkSize * _pOutputStreamMetaEvent->i_SizeOfEventInBytes);
//			+ (iWindowStartEventIndex * _pOutputStreamMetaEvent->i_SizeOfEventInBytes);

	GpuEvent * pInEvent = (GpuEvent*) pInEventBuffer;

	memset(pResultsInEventBufferSegment, 0, _iWorkSize * _pOutputStreamMetaEvent->i_SizeOfEventInBytes);

	// get all matching event for in event from other window buffer and copy them to output event buffer

	// for each events in other window
	int iOtherWindowFillCount  = _iOtherWindowLength - _iOtherRemainingCount;

	if(iWindowStartEventIndex < iOtherWindowFillCount)
	{
		int iWindowEndEventIndex = min(iWindowStartEventIndex + _iWorkSize, iOtherWindowFillCount);

		int iMatchedCount = 0;
		for(int i=iWindowStartEventIndex; i<iWindowEndEventIndex; ++i)
		{
			// get other window event
			char * pOtherWindowEventBuffer = _pOtherEventWindowBuffer + (_pOtherStreamMetaEvent->i_SizeOfEventInBytes * i);
			GpuEvent * pOtherWindowEvent = (GpuEvent*) pOtherWindowEventBuffer;

			// get buffer position for in event matching results
			char * pResultInMatchingEventBuffer = pResultsInEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
			GpuEvent * pResultInMatchingEvent = (GpuEvent*) pResultInMatchingEventBuffer;

			if(pInEvent->i_Sequence > pOtherWindowEvent->i_Sequence &&
					(pInEvent->i_Timestamp - pOtherWindowEvent->i_Timestamp) <= _iWithInTime)
			{

				ExpressionEvalParameters mExpressionParam;
				mExpressionParam.p_OnCompare = _pOnCompareFilter;
				mExpressionParam.a_Meta[0] = _pOtherStreamMetaEvent;
				mExpressionParam.a_Event[0] = pOtherWindowEventBuffer;
				mExpressionParam.a_Meta[1] = _pInputMetaEvent;
				mExpressionParam.a_Event[1] = pInEventBuffer;
				mExpressionParam.i_CurrentIndex = 0;

				bool bOnCompareMatched = Evaluate(mExpressionParam);
				if(bOnCompareMatched)
				{
					// copy output event to buffer - map attributes from input streams to output stream
					pResultInMatchingEvent->i_Type = GpuEvent::CURRENT;
					pResultInMatchingEvent->i_Sequence = pInEvent->i_Sequence;
					pResultInMatchingEvent->i_Timestamp = pInEvent->i_Timestamp;

					for(int m=0; m < _pOutputAttribMappings->i_MappingCount; ++m)
					{
						int iFromStreamIndex = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::STREAM_INDEX];
						int iFromAttrib = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::ATTRIBUTE_INDEX];
						int iTo = _pOutputAttribMappings->p_Mappings[m].to;

						memcpy(
								pResultInMatchingEventBuffer + _pOutputStreamMetaEvent->p_Attributes[iTo].i_Position, // to
								mExpressionParam.a_Event[iFromStreamIndex] + mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Position, // from
								mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Length // size
						);
					}

					iMatchedCount++;
				}
			}
			else
			{
				// cannot continue, last result event for this segment
				pResultInMatchingEvent->i_Type = GpuEvent::RESET;
				break;
			}
		}

		if(iMatchedCount < (iWindowEndEventIndex - iWindowStartEventIndex))
		{
			char * pResultInMatchingEventBuffer = pResultsInEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
			GpuEvent * pResultInMatchingEvent = (GpuEvent*) pResultInMatchingEventBuffer;
			pResultInMatchingEvent->i_Type = GpuEvent::RESET;
		}
	}
	else
	{
		GpuEvent * pResultInMatchingEvent = (GpuEvent*) pResultsInEventBufferSegment;
		pResultInMatchingEvent->i_Type = GpuEvent::RESET;
	}

}

__global__
void ProcessEventsJoinRightTriggerExpireOn(
		char               * _pInputEventBuffer,         // input events buffer
		GpuKernelMetaEvent * _pInputMetaEvent,           // Meta event for input events
		int                  _iInputNumberOfEvents,      // Number of events in input buffer
		char               * _pEventWindowBuffer,        // Event window buffer of this stream
		int                  _iWindowLength,             // Length of current events window
		int                  _iRemainingCount,           // Remaining free slots in Window buffer
		GpuKernelMetaEvent * _pOtherStreamMetaEvent,     // Meta event for other stream
		char               * _pOtherEventWindowBuffer,   // Event window buffer of other stream
		int                  _iOtherWindowLength,        // Length of current events window of other stream
		int                  _iOtherRemainingCount,      // Remaining free slots in Window buffer of other stream
		GpuKernelFilter    * _pOnCompareFilter,          // OnCompare filter buffer - pre-copied at initialization
		uint64_t             _iWithInTime,               // WithIn time in milliseconds
		GpuKernelMetaEvent * _pOutputStreamMetaEvent,    // Meta event for output stream
		char               * _pResultsBuffer,            // Resulting events buffer for this stream
		AttributeMappings  * _pOutputAttribMappings,     // Output event attribute mappings
		int                  _iEventsPerBlock            // number of events allocated per block
)
{
	// avoid out of bound threads
	if(threadIdx.x >= _iEventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _iInputNumberOfEvents / _iEventsPerBlock) && // last thread block
			(threadIdx.x >= _iInputNumberOfEvents % _iEventsPerBlock)) // extra threads
	{
		return;
	}

	// get assigned event
	int iEventIdx = (blockIdx.x * _iEventsPerBlock) + threadIdx.x;

	// output to results buffer [in event, expired event]
	// {other stream event size * other window size} * 2 (for in/exp)
	int iOutputSegmentSize = _pOutputStreamMetaEvent->i_SizeOfEventInBytes * _iOtherWindowLength;

	char * pResultsExpiredEventBufferSegment = _pResultsBuffer + (iOutputSegmentSize * iEventIdx);

	char * pExpiredEventBuffer = NULL;
	GpuEvent * pExpiredEvent = NULL;

	// calculate in/expired event pair for this event

	if(iEventIdx >= _iRemainingCount)
	{
		if(iEventIdx < _iWindowLength)
		{
			// in window buffer
			char * pExpiredOutEventInWindowBuffer = _pEventWindowBuffer + (_pInputMetaEvent->i_SizeOfEventInBytes * (iEventIdx - _iRemainingCount));

			GpuEvent * pWindowEvent = (GpuEvent*) pExpiredOutEventInWindowBuffer;
			if(pWindowEvent->i_Type != GpuEvent::NONE) // if window event is filled
			{
				pExpiredEventBuffer = pExpiredOutEventInWindowBuffer;

			}
			else
			{
				// no expiring event
			}
		}
		else
		{
			// in input event buffer
			char * pExpiredOutEventInInputBuffer = _pInputEventBuffer + (_pInputMetaEvent->i_SizeOfEventInBytes * (iEventIdx - _iWindowLength));
			pExpiredEventBuffer = pExpiredOutEventInInputBuffer;
		}
	}
	else
	{
		// [NULL,inEvent]
		// no expiring event
	}

	if(pExpiredEventBuffer != NULL)
	{

		// get all matching event for in event from other window buffer and copy them to output event buffer

		pExpiredEvent = (GpuEvent*) pExpiredEventBuffer;

		// for each events in other window
		int iOtherWindowFillCount  = _iOtherWindowLength - _iOtherRemainingCount;
		int iMatchedCount = 0;

		// for each events in other window
		for(int i=0; i<iOtherWindowFillCount; ++i)
		{
			// get other window event
			char * pOtherWindowEventBuffer = _pOtherEventWindowBuffer + (_pOtherStreamMetaEvent->i_SizeOfEventInBytes * i);
			GpuEvent * pOtherWindowEvent = (GpuEvent*) pOtherWindowEventBuffer;

			// get buffer position for expire event matching results
			char * pResultExpireMatchingEventBuffer = pResultsExpiredEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
			GpuEvent * pResultExpireMatchingEvent = (GpuEvent*) pResultExpireMatchingEventBuffer;

			if(pExpiredEvent->i_Sequence < pOtherWindowEvent->i_Sequence &&
					(pOtherWindowEvent->i_Timestamp - pExpiredEvent->i_Timestamp) <= _iWithInTime)
			{
				ExpressionEvalParameters mExpressionParam;
				mExpressionParam.p_OnCompare = _pOnCompareFilter;
				mExpressionParam.a_Meta[0] = _pOtherStreamMetaEvent;
				mExpressionParam.a_Event[0] = pOtherWindowEventBuffer;
				mExpressionParam.a_Meta[1] = _pInputMetaEvent;
				mExpressionParam.a_Event[1] = pExpiredEventBuffer;
				mExpressionParam.i_CurrentIndex = 0;

				bool bOnCompareMatched = Evaluate(mExpressionParam);
				if(bOnCompareMatched)
				{
					// copy output event to buffer - map attributes from input streams to output stream
					pResultExpireMatchingEvent->i_Type = GpuEvent::EXPIRED;
					pResultExpireMatchingEvent->i_Sequence = pExpiredEvent->i_Sequence;
					pResultExpireMatchingEvent->i_Timestamp = pExpiredEvent->i_Timestamp;

					for(int m=0; m < _pOutputAttribMappings->i_MappingCount; ++m)
					{
						int iFromStreamIndex = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::STREAM_INDEX];
						int iFromAttrib = _pOutputAttribMappings->p_Mappings[m].from[AttributeMapping::ATTRIBUTE_INDEX];
						int iTo = _pOutputAttribMappings->p_Mappings[m].to;

						memcpy(
								pResultExpireMatchingEventBuffer + _pOutputStreamMetaEvent->p_Attributes[iTo].i_Position, // to
								mExpressionParam.a_Event[iFromStreamIndex] + mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Position, // from
								mExpressionParam.a_Meta[iFromStreamIndex]->p_Attributes[iFromAttrib].i_Length // size
						);
					}

					iMatchedCount++;
				}
			}
			else
			{
				// cannot continue, last result event for this segment
				pResultExpireMatchingEvent->i_Type = GpuEvent::RESET;
				break;
			}
		}

		if(iMatchedCount < iOtherWindowFillCount || iOtherWindowFillCount == 0)
		{
			char * pResultExpireMatchingEventBuffer = pResultsExpiredEventBufferSegment + (_pOutputStreamMetaEvent->i_SizeOfEventInBytes * iMatchedCount);
			GpuEvent * pResultExpireMatchingEvent = (GpuEvent*) pResultExpireMatchingEventBuffer;
			pResultExpireMatchingEvent->i_Type = GpuEvent::RESET;
		}
	}

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
	p_LeftResultEventBuffer(NULL),
	p_RightResultEventBuffer(NULL),
	p_DeviceOnCompareFilter(NULL),
	i_LeftStreamWindowSize(_iLeftWindowSize),
	i_RightStreamWindowSize(_iRightWindowSize),
	i_LeftRemainingCount(_iLeftWindowSize),
	i_RightRemainingCount(_iRightWindowSize),
	i_LeftNumEventPerSegment(0),
	i_RightNumEventPerSegment(0),
	b_LeftFirstKernel(true),
	b_RightFirstKernel(true),
	b_LeftDeviceSet(false),
	b_RightDeviceSet(false),
	i_LeftThreadWorkSize(_iRightWindowSize),
	i_RightThreadWorkSize(_iLeftWindowSize),
	i_LeftThreadWorkCount(0),
	i_RightThreadWorkCount(0),
	i_InitializedStreamCount(0),
	fp_LeftLog(_fpLeftLog),
	fp_RightLog(_fpRightLog)
{
	p_JoinProcessor = (GpuJoinProcessor*) _pProc;
	pthread_mutex_init(&mtx_Lock, NULL);
}

GpuJoinKernel::~GpuJoinKernel()
{
	fprintf(fp_LeftLog, "[GpuJoinKernel] destroy\n");
	fflush(fp_LeftLog);
	fprintf(fp_RightLog, "[GpuJoinKernel] destroy\n");
	fflush(fp_RightLog);

	CUDA_CHECK_RETURN(cudaFree(p_DeviceOnCompareFilter));
	p_DeviceOnCompareFilter = NULL;

	if(p_DeviceOutputAttributeMapping)
	{
		CUDA_CHECK_RETURN(cudaFree(p_DeviceOutputAttributeMapping));
		p_DeviceOutputAttributeMapping = NULL;
	}

	pthread_mutex_destroy(&mtx_Lock);
}

bool GpuJoinKernel::Initialize(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize)
{
	if(_iStreamIndex == 0)
	{
		fprintf(fp_LeftLog, "[GpuJoinKernel] Initialize : StreamIndex=%d LeftTrigger=%d RightTrigger=%d CurrentOn=%d ExpireOn=%d\n",
				_iStreamIndex, p_JoinProcessor->GetLeftTrigger(), p_JoinProcessor->GetRightTrigger(),
				p_JoinProcessor->GetCurrentOn(), p_JoinProcessor->GetExpiredOn());
		fflush(fp_LeftLog);

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

		GpuUtils::PrintThreadInfo("GpuJoinKernel", fp_LeftLog);
	}
	else if(_iStreamIndex == 1)
	{
		fprintf(fp_RightLog, "[GpuJoinKernel] Initialize : StreamIndex=%d LeftTrigger=%d RightTrigger=%d CurrentOn=%d ExpireOn=%d\n",
				_iStreamIndex, p_JoinProcessor->GetLeftTrigger(), p_JoinProcessor->GetRightTrigger(),
				p_JoinProcessor->GetCurrentOn(), p_JoinProcessor->GetExpiredOn());
		fflush(fp_RightLog);

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

		GpuUtils::PrintThreadInfo("GpuJoinKernel", fp_RightLog);
	}

	if(i_InitializedStreamCount == 2)
	{
		fprintf(fp_LeftLog, "[GpuJoinKernel] StreamId=%d Creating result event buffer\n", _iStreamIndex);
		fflush(fp_LeftLog);
		fprintf(fp_RightLog, "[GpuJoinKernel] StreamId=%d Creating result event buffer\n", _iStreamIndex);
		fflush(fp_RightLog);

		p_LeftResultEventBuffer = new GpuStreamEventBuffer("JoinLeftResultEventBuffer", p_LeftContext->GetDeviceId(), p_OutputStreamMeta, fp_LeftLog);
		if(p_JoinProcessor->GetLeftTrigger())
		{
			int iEventCount = 0;
			if(p_JoinProcessor->GetCurrentOn())
			{
				iEventCount += i_RightStreamWindowSize * p_LeftInputEventBuffer->GetMaxEventCount();
				i_LeftNumEventPerSegment = i_RightStreamWindowSize;
			}
			if(p_JoinProcessor->GetExpiredOn())
			{
				iEventCount += i_RightStreamWindowSize * p_LeftInputEventBuffer->GetMaxEventCount();
				i_LeftNumEventPerSegment += i_RightStreamWindowSize;
			}
			p_LeftResultEventBuffer->CreateEventBuffer(iEventCount);
			fprintf(fp_LeftLog, "[GpuJoinKernel] LeftResultEventBuffer created : Size=%d bytes\n", p_LeftResultEventBuffer->GetEventBufferSizeInBytes());
			fflush(fp_LeftLog);
		}
		p_LeftResultEventBuffer->Print();


		p_RightResultEventBuffer = new GpuStreamEventBuffer("JoinRightResultEventBuffer", p_RightContext->GetDeviceId(), p_OutputStreamMeta, fp_RightLog);
		if(p_JoinProcessor->GetRightTrigger())
		{
			int iEventCount = 0;
			if(p_JoinProcessor->GetCurrentOn())
			{
				iEventCount += i_LeftStreamWindowSize * p_RightInputEventBuffer->GetMaxEventCount();
				i_RightNumEventPerSegment = i_LeftStreamWindowSize;
			}
			if(p_JoinProcessor->GetExpiredOn())
			{
				iEventCount += i_LeftStreamWindowSize * p_RightInputEventBuffer->GetMaxEventCount();
				i_RightNumEventPerSegment += i_LeftStreamWindowSize;
			}

			p_RightResultEventBuffer->CreateEventBuffer(iEventCount);
			fprintf(fp_RightLog, "[GpuJoinKernel] RightResultEventBuffer created : Size=%d bytes\n", p_RightResultEventBuffer->GetEventBufferSizeInBytes());
			fflush(fp_RightLog);
		}
		p_RightResultEventBuffer->Print();


		fprintf(fp_LeftLog, "[GpuJoinKernel] Copying OnCompare filter to device \n");
		fflush(fp_LeftLog);
		fprintf(fp_RightLog, "[GpuJoinKernel] Copying OnCompare filter to device \n");
		fflush(fp_RightLog);

		CUDA_CHECK_RETURN(cudaMalloc(
				(void**) &p_DeviceOnCompareFilter,
				sizeof(GpuKernelFilter)));

		GpuKernelFilter * apHostFilters = (GpuKernelFilter *) malloc(sizeof(GpuKernelFilter));


		apHostFilters->i_NodeCount = p_JoinProcessor->i_NodeCount;
		apHostFilters->ap_ExecutorNodes = NULL;

		CUDA_CHECK_RETURN(cudaMalloc(
				(void**) &apHostFilters->ap_ExecutorNodes,
				sizeof(ExecutorNode) * p_JoinProcessor->i_NodeCount));

		CUDA_CHECK_RETURN(cudaMemcpy(
				apHostFilters->ap_ExecutorNodes,
				p_JoinProcessor->ap_ExecutorNodes,
				sizeof(ExecutorNode) * p_JoinProcessor->i_NodeCount,
				cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(
				p_DeviceOnCompareFilter,
				apHostFilters,
				sizeof(GpuKernelFilter),
				cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		CUDA_CHECK_RETURN(cudaThreadSynchronize());

		free(apHostFilters);
		apHostFilters = NULL;

		// copy Output mappings
		if(p_HostOutputAttributeMapping)
		{
			fprintf(fp_LeftLog, "[GpuJoinKernel] Copying AttributeMappings to device \n");
			fflush(fp_LeftLog);
			fprintf(fp_RightLog, "[GpuJoinKernel] Copying AttributeMappings to device \n");
			fflush(fp_RightLog);

			fprintf(fp_LeftLog, "[GpuJoinKernel] AttributeMapCount : %d \n", p_HostOutputAttributeMapping->i_MappingCount);
			fprintf(fp_RightLog, "[GpuJoinKernel] AttributeMapCount : %d \n", p_HostOutputAttributeMapping->i_MappingCount);
			for(int c=0; c<p_HostOutputAttributeMapping->i_MappingCount; ++c)
			{
				fprintf(fp_LeftLog, "[GpuJoinKernel] Map : Form [Stream=%d, Attrib=%d] To [Attrib=%d] \n",
						p_HostOutputAttributeMapping->p_Mappings[c].from[AttributeMapping::STREAM_INDEX],
						p_HostOutputAttributeMapping->p_Mappings[c].from[AttributeMapping::ATTRIBUTE_INDEX],
						p_HostOutputAttributeMapping->p_Mappings[c].to);

				fprintf(fp_RightLog, "[GpuJoinKernel] Map : Form [Stream=%d, Attrib=%d] To [Attrib=%d] \n",
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

		if(p_JoinProcessor->GetThreadWorkSize() != 0)
		{
			i_LeftThreadWorkSize = p_JoinProcessor->GetThreadWorkSize();
			i_RightThreadWorkSize = p_JoinProcessor->GetThreadWorkSize();
		}

		i_LeftThreadWorkCount = ceil((float)i_RightStreamWindowSize / i_LeftThreadWorkSize);
		i_RightThreadWorkCount = ceil((float)i_LeftStreamWindowSize / i_RightThreadWorkSize);

		fprintf(fp_LeftLog, "[GpuJoinKernel] LeftThreadWorkCount=%d RightThreadWorkCount=%d\n", i_LeftThreadWorkCount, i_RightThreadWorkCount);
		fflush(fp_LeftLog);
		fprintf(fp_RightLog, "[GpuJoinKernel] LeftThreadWorkCount=%d RightThreadWorkCount=%d\n", i_LeftThreadWorkCount, i_RightThreadWorkCount);
		fflush(fp_RightLog);

		fprintf(fp_LeftLog, "[GpuJoinKernel] Initialization complete\n");
		fflush(fp_LeftLog);
		fprintf(fp_RightLog, "[GpuJoinKernel] Initialization complete\n");
		fflush(fp_RightLog);
	}

	return true;
}

void GpuJoinKernel::Process(int _iStreamIndex, int & _iNumEvents)
{
	if(_iStreamIndex == 0)
	{
		ProcessLeftStream(_iStreamIndex, _iNumEvents);
	}
	else if(_iStreamIndex == 1)
	{
		ProcessRightStream(_iStreamIndex, _iNumEvents);
	}
}

void GpuJoinKernel::ProcessLeftStream(int _iStreamIndex, int & _iNumEvents)
{
#if GPU_DEBUG >= GPU_DEBUG_LEVEL_DEBUG
	fprintf(fp_LeftLog, "[GpuJoinKernel] ProcessLeftStream : StreamIndex=%d EventCount=%d\n", _iStreamIndex, _iNumEvents);
	GpuUtils::PrintThreadInfo("GpuJoinKernel::ProcessLeftStream", fp_LeftLog);
	fflush(fp_LeftLog);
#endif

	if(!b_LeftDeviceSet)
	{
		GpuCudaHelper::SelectDevice(i_DeviceId, "GpuJoinKernel::Left", fp_LeftLog);
		b_LeftDeviceSet = true;
	}

#ifdef KERNEL_TIME
	sdkStartTimer(&p_StopWatch);
#endif

	if(b_LeftFirstKernel)
	{
		p_LeftInputEventBuffer->CopyToDevice(true);
	}

	// call entry kernel
	int numBlocksX = ceil((float)_iNumEvents * i_LeftThreadWorkCount / (float)i_ThreadBlockSize);
	int numBlocksY = 1;
	dim3 numBlocks = dim3(numBlocksX, numBlocksY);
	dim3 numThreads = dim3(i_ThreadBlockSize, 1);

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(fp_LeftLog, "[GpuJoinKernel] ProcessLeftStream : Invoke kernel Blocks(%d,%d) Threads(%d,%d)\n", numBlocksX, numBlocksY, i_ThreadBlockSize, 1);
	fprintf(fp_LeftLog, "[GpuJoinKernel] ProcessLeftStream : NumEvents=%d LeftWindow=(%d/%d) RightWindow=(%d/%d) WithIn=%llu\n",
			_iNumEvents, i_LeftRemainingCount, i_LeftStreamWindowSize, i_RightRemainingCount, i_RightStreamWindowSize,
			p_JoinProcessor->GetWithInTimeMilliSeconds());
	fflush(fp_LeftLog);
#endif

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_TRACE
	GpuUtils::PrintByteBuffer(p_LeftInputEventBuffer->GetHostEventBuffer(), _iNumEvents,
			p_LeftInputEventBuffer->GetHostMetaEvent(), "GpuJoinKernel:LeftInputBuffer", fp_LeftLog);

	p_LeftWindowEventBuffer->CopyToHost(false);
	GpuUtils::PrintByteBuffer(p_LeftWindowEventBuffer->GetHostEventBuffer(), (i_LeftStreamWindowSize - i_LeftRemainingCount),
			p_LeftWindowEventBuffer->GetHostMetaEvent(), "GpuJoinKernel:LeftWindowBuffer", fp_LeftLog);

	p_RightWindowEventBuffer->CopyToHost(false);
	GpuUtils::PrintByteBuffer(p_RightWindowEventBuffer->GetHostEventBuffer(), (i_RightStreamWindowSize - i_RightRemainingCount),
			p_RightWindowEventBuffer->GetHostMetaEvent(), "GpuJoinKernel:RightWindowBuffer", fp_LeftLog);

	fflush(fp_LeftLog);
#endif

//	char               * _pInputEventBuffer,         // input events buffer
//	GpuKernelMetaEvent * _pInputMetaEvent,           // Meta event for input events
//	int                  _iInputNumberOfEvents,      // Number of events in input buffer
//	char               * _pEventWindowBuffer,        // Event window buffer of this stream
//	int                  _iWindowLength,             // Length of current events window
//	int                  _iRemainingCount,           // Remaining free slots in Window buffer
//	GpuKernelMetaEvent * _pOtherStreamMetaEvent,     // Meta event for other stream
//	char               * _pOtherEventWindowBuffer,   // Event window buffer of other stream
//	int                  _iOtherWindowLength,        // Length of current events window of other stream
//	int                  _iOtherRemainingCount,      // Remaining free slots in Window buffer of other stream
//	GpuKernelFilter    * _pOnCompareFilter,          // OnCompare filter buffer - pre-copied at initialization
//	int                  _iWithInTime,               // WithIn time in milliseconds
//	GpuKernelMetaEvent * _pOutputStreamMetaEvent,    // Meta event for output stream
//	char               * _pResultsBuffer,            // Resulting events buffer for this stream
//	AttributeMappings  * _pOutputAttribMappings,     // Output event attribute mappings
//	int                  _iEventsPerBlock            // number of events allocated per block

	if(p_JoinProcessor->GetLeftTrigger())
	{

		if(p_JoinProcessor->GetCurrentOn() && p_JoinProcessor->GetExpiredOn())
		{
			ProcessEventsJoinLeftTriggerAllOn<<<numBlocks, numThreads>>>(
					p_LeftInputEventBuffer->GetDeviceEventBuffer(),
					p_LeftInputEventBuffer->GetDeviceMetaEvent(),
					_iNumEvents,
					p_LeftWindowEventBuffer->GetDeviceEventBuffer(),
					i_LeftStreamWindowSize,
					i_LeftRemainingCount,
					p_RightInputEventBuffer->GetDeviceMetaEvent(),
					p_RightWindowEventBuffer->GetDeviceEventBuffer(),
					i_RightStreamWindowSize,
					i_RightRemainingCount,
					p_DeviceOnCompareFilter,
					p_JoinProcessor->GetWithInTimeMilliSeconds(),
					p_LeftResultEventBuffer->GetDeviceMetaEvent(),
					p_LeftResultEventBuffer->GetDeviceEventBuffer(),
					p_DeviceOutputAttributeMapping,
					i_ThreadBlockSize
			);
		}
		else if(p_JoinProcessor->GetCurrentOn())
		{
			ProcessEventsJoinLeftTriggerCurrentOn<<<numBlocks, numThreads>>>(
					p_LeftInputEventBuffer->GetDeviceEventBuffer(),
					p_LeftInputEventBuffer->GetDeviceMetaEvent(),
					_iNumEvents,
					p_LeftWindowEventBuffer->GetDeviceEventBuffer(),
					i_LeftStreamWindowSize,
					i_LeftRemainingCount,
					p_RightInputEventBuffer->GetDeviceMetaEvent(),
					p_RightWindowEventBuffer->GetDeviceEventBuffer(),
					i_RightStreamWindowSize,
					i_RightRemainingCount,
					p_DeviceOnCompareFilter,
					p_JoinProcessor->GetWithInTimeMilliSeconds(),
					p_LeftResultEventBuffer->GetDeviceMetaEvent(),
					p_LeftResultEventBuffer->GetDeviceEventBuffer(),
					p_DeviceOutputAttributeMapping,
					i_ThreadBlockSize,
					i_LeftThreadWorkSize
			);
		}
		else if(p_JoinProcessor->GetExpiredOn())
		{
			ProcessEventsJoinLeftTriggerExpiredOn<<<numBlocks, numThreads>>>(
					p_LeftInputEventBuffer->GetDeviceEventBuffer(),
					p_LeftInputEventBuffer->GetDeviceMetaEvent(),
					_iNumEvents,
					p_LeftWindowEventBuffer->GetDeviceEventBuffer(),
					i_LeftStreamWindowSize,
					i_LeftRemainingCount,
					p_RightInputEventBuffer->GetDeviceMetaEvent(),
					p_RightWindowEventBuffer->GetDeviceEventBuffer(),
					i_RightStreamWindowSize,
					i_RightRemainingCount,
					p_DeviceOnCompareFilter,
					p_JoinProcessor->GetWithInTimeMilliSeconds(),
					p_LeftResultEventBuffer->GetDeviceMetaEvent(),
					p_LeftResultEventBuffer->GetDeviceEventBuffer(),
					p_DeviceOutputAttributeMapping,
					i_ThreadBlockSize
			);
		}

	}

	numBlocksX = ceil((float)_iNumEvents / (float)i_ThreadBlockSize);
	numBlocks = dim3(numBlocksX, numBlocksY);

	// we need to synchronize processing of JoinKernel as only one batch of events can be there at a time
	pthread_mutex_lock(&mtx_Lock);

//	char               * _pInputEventBuffer,     // original input events buffer
//	int                  _iNumberOfEvents,       // Number of events in input buffer (matched + not matched)
//	char               * _pEventWindowBuffer,    // Event window buffer
//	int                  _iWindowLength,         // Length of current events window
//	int                  _iRemainingCount,       // Remaining free slots in Window buffer
//	int                  _iMaxEventCount,        // used for setting results array
//	int                  _iSizeOfEvent,          // Size of an event
//	int                  _iEventsPerBlock        // number of events allocated per block

	JoinSetWindowState<<<numBlocks, numThreads>>>(
			p_LeftInputEventBuffer->GetDeviceEventBuffer(),
			_iNumEvents,
			p_LeftWindowEventBuffer->GetDeviceEventBuffer(),
			i_LeftStreamWindowSize,
			i_LeftRemainingCount,
			p_LeftInputEventBuffer->GetMaxEventCount(),
			p_LeftInputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes,
			i_ThreadBlockSize
	);

	if(b_LastKernel)
	{
		p_LeftResultEventBuffer->CopyToHost(true);
#if GPU_DEBUG >= GPU_DEBUG_LEVEL_DEBUG
	fprintf(fp_LeftLog, "[GpuJoinKernel] Results copied \n");
	fflush(fp_LeftLog);
#endif
	}

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(fp_LeftLog, "[GpuJoinKernel] Kernel complete \n");
	fflush(fp_LeftLog);
#endif



#ifdef KERNEL_TIME
	sdkStopTimer(&p_StopWatch);
	float fElapsed = sdkGetTimerValue(&p_StopWatch);
	fprintf(fp_LeftLog, "[GpuJoinKernel] Stats : Elapsed=%f ms\n", fElapsed);
	fflush(fp_LeftLog);
	lst_ElapsedTimes.push_back(fElapsed);
	sdkResetTimer(&p_StopWatch);
#endif


	if(_iNumEvents > i_LeftRemainingCount)
	{
		i_LeftRemainingCount = 0;
	}
	else
	{
		i_LeftRemainingCount -= _iNumEvents;
	}

	pthread_mutex_unlock(&mtx_Lock);

	if(!p_JoinProcessor->GetLeftTrigger())
	{
		_iNumEvents = 0;
	}
	else
	{
		_iNumEvents = _iNumEvents * i_LeftNumEventPerSegment;
	}


#if GPU_DEBUG >= GPU_DEBUG_LEVEL_DEBUG
	GpuUtils::PrintByteBuffer(p_LeftResultEventBuffer->GetHostEventBuffer(), _iNumEvents,
			p_LeftResultEventBuffer->GetHostMetaEvent(), "GpuJoinKernel:LeftResultEventBuffer", fp_LeftLog);
	fflush(fp_LeftLog);
#endif

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_TRACE
	p_LeftWindowEventBuffer->CopyToHost(true);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	GpuUtils::PrintByteBuffer(p_LeftWindowEventBuffer->GetHostEventBuffer(), (i_LeftStreamWindowSize - i_LeftRemainingCount),
			p_LeftWindowEventBuffer->GetHostMetaEvent(), "GpuJoinKernel:LeftWindowBuffer", fp_LeftLog);
	fflush(fp_LeftLog);
#endif
}

void GpuJoinKernel::ProcessRightStream(int _iStreamIndex, int & _iNumEvents)
{
#if GPU_DEBUG >= GPU_DEBUG_LEVEL_DEBUG
	fprintf(fp_RightLog, "[GpuJoinKernel] ProcessRightStream : StreamIndex=%d EventCount=%d\n", _iStreamIndex, _iNumEvents);
	GpuUtils::PrintThreadInfo("GpuJoinKernel::ProcessRightStream", fp_RightLog);
	fflush(fp_RightLog);
#endif

	if(!b_RightDeviceSet)
	{
		GpuCudaHelper::SelectDevice(i_DeviceId, "GpuJoinKernel::Right", fp_RightLog);
		b_RightDeviceSet = true;
	}

#ifdef KERNEL_TIME
	sdkStartTimer(&p_StopWatch);
#endif

	if(b_RightFirstKernel)
	{
		p_RightInputEventBuffer->CopyToDevice(true);
	}

	// call entry kernel
	int numBlocksX = ceil((float)_iNumEvents * i_RightThreadWorkCount / (float)i_ThreadBlockSize);
	int numBlocksY = 1;
	dim3 numBlocks = dim3(numBlocksX, numBlocksY);
	dim3 numThreads = dim3(i_ThreadBlockSize, 1);

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(fp_RightLog, "[GpuJoinKernel] ProcessRightStream : Invoke kernel Blocks(%d,%d) Threads(%d,%d)\n", numBlocksX, numBlocksY, i_ThreadBlockSize, 1);
	fprintf(fp_RightLog, "[GpuJoinKernel] ProcessLeftStream : NumEvents=%d LeftWindow=(%d/%d) RightWindow=(%d/%d) WithIn=%llu\n",
			_iNumEvents, i_LeftRemainingCount, i_LeftStreamWindowSize, i_RightRemainingCount, i_RightStreamWindowSize,
			p_JoinProcessor->GetWithInTimeMilliSeconds());
	fflush(fp_RightLog);
#endif

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_TRACE
	GpuUtils::PrintByteBuffer(p_RightInputEventBuffer->GetHostEventBuffer(), _iNumEvents,
			p_RightInputEventBuffer->GetHostMetaEvent(), "GpuJoinKernel:RightInputBuffer", fp_RightLog);

	p_LeftWindowEventBuffer->CopyToHost(false);
	GpuUtils::PrintByteBuffer(p_LeftWindowEventBuffer->GetHostEventBuffer(), (i_LeftStreamWindowSize - i_LeftRemainingCount),
			p_LeftWindowEventBuffer->GetHostMetaEvent(), "GpuJoinKernel:LeftWindowBuffer", fp_RightLog);

	p_RightWindowEventBuffer->CopyToHost(false);
	GpuUtils::PrintByteBuffer(p_RightWindowEventBuffer->GetHostEventBuffer(), (i_RightStreamWindowSize - i_RightRemainingCount),
			p_RightWindowEventBuffer->GetHostMetaEvent(), "GpuJoinKernel:RightWindowBuffer", fp_RightLog);

	fflush(fp_RightLog);
#endif

//	char               * _pInputEventBuffer,         // input events buffer
//	GpuKernelMetaEvent * _pInputMetaEvent,           // Meta event for input events
//	int                  _iInputNumberOfEvents,      // Number of events in input buffer
//	char               * _pEventWindowBuffer,        // Event window buffer of this stream
//	int                  _iWindowLength,             // Length of current events window
//	int                  _iRemainingCount,           // Remaining free slots in Window buffer
//	GpuKernelMetaEvent * _pOtherStreamMetaEvent,     // Meta event for other stream
//	char               * _pOtherEventWindowBuffer,   // Event window buffer of other stream
//	int                  _iOtherWindowLength,        // Length of current events window of other stream
//	int                  _iOtherRemainingCount,      // Remaining free slots in Window buffer of other stream
//	GpuKernelFilter    * _pOnCompareFilter,          // OnCompare filter buffer - pre-copied at initialization
//	int                  _iWithInTime,               // WithIn time in milliseconds
//	GpuKernelMetaEvent * _pOutputStreamMetaEvent,    // Meta event for output stream
//	char               * _pResultsBuffer,            // Resulting events buffer for this stream
//	AttributeMappings  * _pOutputAttribMappings,     // Output event attribute mappings
//	int                  _iEventsPerBlock            // number of events allocated per block

	if(p_JoinProcessor->GetRightTrigger())
	{
		if(p_JoinProcessor->GetCurrentOn() && p_JoinProcessor->GetExpiredOn())
		{
			ProcessEventsJoinRightTriggerAllOn<<<numBlocks, numThreads>>>(
					p_RightInputEventBuffer->GetDeviceEventBuffer(),
					p_RightInputEventBuffer->GetDeviceMetaEvent(),
					_iNumEvents,
					p_RightWindowEventBuffer->GetDeviceEventBuffer(),
					i_RightStreamWindowSize,
					i_RightRemainingCount,
					p_LeftInputEventBuffer->GetDeviceMetaEvent(),
					p_LeftWindowEventBuffer->GetDeviceEventBuffer(),
					i_LeftStreamWindowSize,
					i_LeftRemainingCount,
					p_DeviceOnCompareFilter,
					p_JoinProcessor->GetWithInTimeMilliSeconds(),
					p_RightResultEventBuffer->GetDeviceMetaEvent(),
					p_RightResultEventBuffer->GetDeviceEventBuffer(),
					p_DeviceOutputAttributeMapping,
					i_ThreadBlockSize
			);
		}
		else if(p_JoinProcessor->GetCurrentOn())
		{
			ProcessEventsJoinRightTriggerCurrentOn<<<numBlocks, numThreads>>>(
					p_RightInputEventBuffer->GetDeviceEventBuffer(),
					p_RightInputEventBuffer->GetDeviceMetaEvent(),
					_iNumEvents,
					p_RightWindowEventBuffer->GetDeviceEventBuffer(),
					i_RightStreamWindowSize,
					i_RightRemainingCount,
					p_LeftInputEventBuffer->GetDeviceMetaEvent(),
					p_LeftWindowEventBuffer->GetDeviceEventBuffer(),
					i_LeftStreamWindowSize,
					i_LeftRemainingCount,
					p_DeviceOnCompareFilter,
					p_JoinProcessor->GetWithInTimeMilliSeconds(),
					p_RightResultEventBuffer->GetDeviceMetaEvent(),
					p_RightResultEventBuffer->GetDeviceEventBuffer(),
					p_DeviceOutputAttributeMapping,
					i_ThreadBlockSize,
					i_RightThreadWorkSize
			);
		}
		else if(p_JoinProcessor->GetExpiredOn())
		{
			ProcessEventsJoinRightTriggerExpireOn<<<numBlocks, numThreads>>>(
					p_RightInputEventBuffer->GetDeviceEventBuffer(),
					p_RightInputEventBuffer->GetDeviceMetaEvent(),
					_iNumEvents,
					p_RightWindowEventBuffer->GetDeviceEventBuffer(),
					i_RightStreamWindowSize,
					i_RightRemainingCount,
					p_LeftInputEventBuffer->GetDeviceMetaEvent(),
					p_LeftWindowEventBuffer->GetDeviceEventBuffer(),
					i_LeftStreamWindowSize,
					i_LeftRemainingCount,
					p_DeviceOnCompareFilter,
					p_JoinProcessor->GetWithInTimeMilliSeconds(),
					p_RightResultEventBuffer->GetDeviceMetaEvent(),
					p_RightResultEventBuffer->GetDeviceEventBuffer(),
					p_DeviceOutputAttributeMapping,
					i_ThreadBlockSize
			);
		}

	}

	numBlocksX = ceil((float)_iNumEvents / (float)i_ThreadBlockSize);
	numBlocks = dim3(numBlocksX, numBlocksY);

	// we need to synchronize processing of JoinKernel as only one batch of events can be there at a time
	pthread_mutex_lock(&mtx_Lock);

//	char               * _pInputEventBuffer,     // original input events buffer
//	int                  _iNumberOfEvents,       // Number of events in input buffer (matched + not matched)
//	char               * _pEventWindowBuffer,    // Event window buffer
//	int                  _iWindowLength,         // Length of current events window
//	int                  _iRemainingCount,       // Remaining free slots in Window buffer
//	int                  _iMaxEventCount,        // used for setting results array
//	int                  _iSizeOfEvent,          // Size of an event
//	int                  _iEventsPerBlock        // number of events allocated per block

	JoinSetWindowState<<<numBlocks, numThreads>>>(
			p_RightInputEventBuffer->GetDeviceEventBuffer(),
			_iNumEvents,
			p_RightWindowEventBuffer->GetDeviceEventBuffer(),
			i_RightStreamWindowSize,
			i_RightRemainingCount,
			p_RightInputEventBuffer->GetMaxEventCount(),
			p_RightInputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes,
			i_ThreadBlockSize
	);

	if(b_LastKernel)
	{
		p_RightResultEventBuffer->CopyToHost(true);
#if GPU_DEBUG >= GPU_DEBUG_LEVEL_DEBUG
	fprintf(fp_RightLog, "[GpuJoinKernel] Results copied \n");
	fflush(fp_RightLog);
#endif
	}

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(fp_RightLog, "[GpuJoinKernel] Kernel complete \n");
	fflush(fp_RightLog);
#endif



#ifdef KERNEL_TIME
	sdkStopTimer(&p_StopWatch);
	float fElapsed = sdkGetTimerValue(&p_StopWatch);
	fprintf(fp_RightLog, "[GpuJoinKernel] Stats : Elapsed=%f ms\n", fElapsed);
	fflush(fp_RightLog);
	lst_ElapsedTimes.push_back(fElapsed);
	sdkResetTimer(&p_StopWatch);
#endif


	if(_iNumEvents > i_RightRemainingCount)
	{
		i_RightRemainingCount = 0;
	}
	else
	{
		i_RightRemainingCount -= _iNumEvents;
	}

	pthread_mutex_unlock(&mtx_Lock);

	if(!p_JoinProcessor->GetRightTrigger())
	{
		_iNumEvents = 0;
	}
	else
	{
		_iNumEvents = _iNumEvents * i_RightNumEventPerSegment;
	}


#if GPU_DEBUG >= GPU_DEBUG_LEVEL_DEBUG
	GpuUtils::PrintByteBuffer(p_RightResultEventBuffer->GetHostEventBuffer(), _iNumEvents,
			p_RightResultEventBuffer->GetHostMetaEvent(), "GpuJoinKernel:RightResultEventBuffer", fp_RightLog);
	fflush(fp_RightLog);
#endif

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_TRACE
	p_RightWindowEventBuffer->CopyToHost(true);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	GpuUtils::PrintByteBuffer(p_RightWindowEventBuffer->GetHostEventBuffer(), (i_RightStreamWindowSize - i_RightRemainingCount),
			p_RightWindowEventBuffer->GetHostMetaEvent(), "GpuJoinKernel:RightWindowBuffer", fp_RightLog);
	fflush(fp_RightLog);
#endif
}

char * GpuJoinKernel::GetResultEventBuffer()
{
	return NULL;
}

int GpuJoinKernel::GetResultEventBufferSize()
{
	return 0;
}

char * GpuJoinKernel::GetLeftResultEventBuffer()
{
	return p_LeftResultEventBuffer->GetHostEventBuffer();
}

int GpuJoinKernel::GetLeftResultEventBufferSize()
{
	return p_LeftResultEventBuffer->GetEventBufferSizeInBytes();
}

char * GpuJoinKernel::GetRightResultEventBuffer()
{
	return p_RightResultEventBuffer->GetHostEventBuffer();
}

int GpuJoinKernel::GetRightResultEventBufferSize()
{
	return p_RightResultEventBuffer->GetEventBufferSizeInBytes();
}

}

#endif
