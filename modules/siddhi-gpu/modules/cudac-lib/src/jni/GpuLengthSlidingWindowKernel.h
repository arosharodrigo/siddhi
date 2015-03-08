/*
 * GpuLengthSlidingWindowKernel.h
 *
 *  Created on: Jan 28, 2015
 *      Author: prabodha
 */

#ifndef GPULENGTHSLIDINGWINDOWKERNEL_H_
#define GPULENGTHSLIDINGWINDOWKERNEL_H_

#include <stdio.h>
#include "GpuKernel.h"

namespace SiddhiGpu
{

class GpuProcessor;
class GpuProcessorContext;
class GpuMetaEvent;
class GpuStreamEventBuffer;
class GpuIntBuffer;

typedef struct LengthSlidingWindowKernelParameters
{
	char               * p_InputEventBuffer;     // original input events buffer
	GpuKernelMetaEvent * p_InputEventMeta;      // Input event meta
	int                  i_SizeOfInputEvent;     // Input event meta
	char               * p_EventWindowBuffer;    // Event window buffer
	int                  i_WindowLength;         // Length of current events window
	char               * p_ResultsBuffer;        // Resulting events buffer
	GpuKernelMetaEvent * p_OutputEventMeta;      // Output event meta
	AttributeMappings  * p_AttributeMapping;     // Attribute mapping
	int                  i_EventsPerBlock;       // number of events allocated per block
}LengthSlidingWindowKernelParameters;

class GpuLengthSlidingWindowFirstKernel : public GpuKernel
{
public:
	GpuLengthSlidingWindowFirstKernel(GpuProcessor * _pProc, GpuProcessorContext * _pContext, int _iThreadBlockSize,
			int _iWindowSize, FILE * _fPLog);
	~GpuLengthSlidingWindowFirstKernel();

	bool Initialize(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize);
	void Process(int _iStreamIndex, int & _iNumEvents);
	char * GetResultEventBuffer();
	int GetResultEventBufferSize();

private:
	GpuProcessorContext * p_Context;

	GpuStreamEventBuffer * p_InputEventBuffer;
	GpuStreamEventBuffer * p_ResultEventBuffer;
	GpuStreamEventBuffer * p_WindowEventBuffer;

	LengthSlidingWindowKernelParameters * p_DeviceParameters;

	bool b_DeviceSet;
	int i_WindowSize;
	int i_RemainingCount;
};

class GpuLengthSlidingWindowFilterKernel : public GpuKernel
{
public:
	GpuLengthSlidingWindowFilterKernel(GpuProcessor * _pProc, GpuProcessorContext * _pContext, int _iThreadBlockSize,
			int _iWindowSize, FILE * _fPLog);
	~GpuLengthSlidingWindowFilterKernel();

	bool Initialize(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize);
	void Process(int _iStreamIndex, int & _iNumEvents);
	char * GetResultEventBuffer();
	int GetResultEventBufferSize();

private:
	GpuProcessorContext * p_Context;

	GpuStreamEventBuffer * p_InputEventBuffer;
	GpuStreamEventBuffer * p_ResultEventBuffer;
	GpuStreamEventBuffer * p_WindowEventBuffer;

	LengthSlidingWindowKernelParameters * p_DeviceParameters;

	bool b_DeviceSet;
	int i_WindowSize;
	int i_RemainingCount;
};

}


#endif /* GPULENGTHSLIDINGWINDOWKERNEL_H_ */
