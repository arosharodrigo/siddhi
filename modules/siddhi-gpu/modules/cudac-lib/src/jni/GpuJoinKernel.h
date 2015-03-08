/*
 * GpuJoinKernel.h
 *
 *  Created on: Jan 31, 2015
 *      Author: prabodha
 */

#ifndef GPUJOINKERNEL_H_
#define GPUJOINKERNEL_H_

#include <stdio.h>
#include "GpuKernelDataTypes.h"
#include "GpuKernel.h"

namespace SiddhiGpu
{

class GpuMetaEvent;
class GpuProcessor;
class GpuProcessorContext;
class GpuStreamEventBuffer;
class GpuWindowEventBuffer;
class GpuIntBuffer;
class GpuRawByteBuffer;
class GpuJoinProcessor;

typedef struct JoinKernelParameters
{
	char               * p_InputEventBuffer;         // input events buffer
	GpuKernelMetaEvent * p_InputMetaEvent;           // Meta event for input events
//	int                  i_InputNumberOfEvents;      // Number of events in input buffer
	char               * p_EventWindowBuffer;        // Event window buffer of this stream
	int                  i_WindowLength;             // Length of current events window
//	int                  i_RemainingCount;           // Remaining free slots in Window buffer
	GpuKernelMetaEvent * p_OtherStreamMetaEvent;     // Meta event for other stream
	char               * p_OtherEventWindowBuffer;   // Event window buffer of other stream
	int                  i_OtherWindowLength;        // Length of current events window of other stream
//	int                  i_OtherRemainingCount;      // Remaining free slots in Window buffer of other stream
	GpuKernelFilter    * p_OnCompareFilter;          // OnCompare filter buffer - pre-copied at initialization
	uint64_t             i_WithInTime;               // WithIn time in milliseconds
	GpuKernelMetaEvent * p_OutputStreamMetaEvent;    // Meta event for output stream
	char               * p_ResultsBuffer;            // Resulting events buffer for this stream
	AttributeMappings  * p_OutputAttribMappings;     // Output event attribute mappings
	int                  i_EventsPerBlock;           // number of events allocated per block
	int                  i_WorkSize;                  // Number of events in window process by this kernel
} JoinKernelParameters;

class GpuJoinKernel : public GpuKernel
{
public:
	GpuJoinKernel(GpuProcessor * _pProc, GpuProcessorContext * _pLeftContext, GpuProcessorContext * _pRightContext,
			int _iThreadBlockSize, int _iLeftWindowSize, int _iRightWindowSize, FILE * _fpLeftLog, FILE * _fpRightLog);
	~GpuJoinKernel();

	bool Initialize(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize);
	void Process(int _iStreamIndex, int &_iNumEvents);
	char * GetResultEventBuffer();
	int GetResultEventBufferSize();

	int GetLeftInputEventBufferIndex() { return i_LeftInputBufferIndex; }
	void SetLeftInputEventBufferIndex(int _iIndex) { i_LeftInputBufferIndex = _iIndex; }
	int GetRightInputEventBufferIndex() { return i_RightInputBufferIndex; }
	void SetRightInputEventBufferIndex(int _iIndex) { i_RightInputBufferIndex = _iIndex; }

	char * GetLeftResultEventBuffer();
	int GetLeftResultEventBufferSize();
	char * GetRightResultEventBuffer();
	int GetRightResultEventBufferSize();

	void SetLeftFirstKernel(bool _bSet) { b_LeftFirstKernel = _bSet; }
	void SetRightFirstKernel(bool _bSet) { b_RightFirstKernel = _bSet; }

private:
	void ProcessLeftStream(int _iStreamIndex, int & _iNumEvents);
	void ProcessRightStream(int _iStreamIndex, int & _iNumEvents);

	GpuJoinProcessor * p_JoinProcessor;

	GpuProcessorContext * p_LeftContext;
	GpuProcessorContext * p_RightContext;

	int i_LeftInputBufferIndex;
	int i_RightInputBufferIndex;

	GpuStreamEventBuffer * p_LeftInputEventBuffer;
	GpuStreamEventBuffer * p_RightInputEventBuffer;

	GpuWindowEventBuffer * p_LeftWindowEventBuffer;
	GpuWindowEventBuffer * p_RightWindowEventBuffer;

	GpuStreamEventBuffer * p_LeftResultEventBuffer;
	GpuStreamEventBuffer * p_RightResultEventBuffer;

	GpuKernelFilter * p_DeviceOnCompareFilter;

	JoinKernelParameters * p_DeviceParametersLeft;
	JoinKernelParameters * p_DeviceParametersRight;

	int i_LeftStreamWindowSize;
	int i_RightStreamWindowSize;
//	int i_LeftRemainingCount;
//	int i_RightRemainingCount;

	int i_LeftNumEventPerSegment;
	int i_RightNumEventPerSegment;

	bool b_LeftFirstKernel;
	bool b_RightFirstKernel;

	bool b_LeftDeviceSet;
	bool b_RightDeviceSet;

	int i_LeftThreadWorkSize;
	int i_RightThreadWorkSize;
	int i_LeftThreadWorkerCount;
	int i_RightThreadWorkerCount;

	int i_InitializedStreamCount;
	FILE * fp_LeftLog;
	FILE * fp_RightLog;

	pthread_mutex_t mtx_Lock;
};

}


#endif /* GPUJOINKERNEL_H_ */
