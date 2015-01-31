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

class GpuLengthSlidingWindowFirstKernel : public GpuKernel
{
public:
	GpuLengthSlidingWindowFirstKernel(GpuProcessor * _pProc, GpuProcessorContext * _pContext, int _iThreadBlockSize,
			int _iWindowSize, FILE * _fPLog);
	~GpuLengthSlidingWindowFirstKernel();

	bool Initialize(GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize);
	void Process(int _iNumEvents, bool _bLast);
	char * GetResultEventBuffer();
	int GetResultEventBufferSize();

private:
	GpuProcessorContext * p_Context;

	GpuStreamEventBuffer * p_InputEventBuffer;
	GpuStreamEventBuffer * p_ResultEventBuffer;
	GpuStreamEventBuffer * p_WindowEventBuffer;

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

	bool Initialize(GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize);
	void Process(int _iNumEvents, bool _bLast);
	char * GetResultEventBuffer();
	int GetResultEventBufferSize();

private:
	GpuProcessorContext * p_Context;

	GpuIntBuffer * p_InputEventBuffer;
	GpuStreamEventBuffer * p_ResultEventBuffer;
	GpuStreamEventBuffer * p_OriginalEventBuffer;
	GpuStreamEventBuffer * p_WindowEventBuffer;

	bool b_DeviceSet;
	int i_WindowSize;
	int i_RemainingCount;
};

}


#endif /* GPULENGTHSLIDINGWINDOWKERNEL_H_ */
