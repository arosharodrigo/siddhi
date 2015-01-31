/*
 * GpuLengthSlidingWindowProcessor.h
 *
 *  Created on: Jan 20, 2015
 *      Author: prabodha
 */

#ifndef GPULENGTHSLIDINGWINDOWPROCESSOR_H_
#define GPULENGTHSLIDINGWINDOWPROCESSOR_H_

#include <stdio.h>
#include "GpuProcessor.h"

namespace SiddhiGpu
{

class GpuMetaEvent;
class GpuProcessorContext;
class GpuLengthSlidingWindowFilterKernel;

class GpuLengthSlidingWindowProcessor : public GpuProcessor
{
public:
	GpuLengthSlidingWindowProcessor(int _iWindowSize);
	virtual ~GpuLengthSlidingWindowProcessor();

	void Configure(GpuProcessor * _pPrevProcessor, GpuProcessorContext * _pContext, FILE * _fpLog);
	void Init(GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize);
	void Process(int _iNumEvents);
	void Print(FILE * _fp);
	GpuProcessor * Clone();
	int GetResultEventBufferIndex();
	char * GetResultEventBuffer();
	int GetResultEventBufferSize();

	void Print() { Print(stdout); }

	int GetWindowSize() { return i_WindowSize; }

private:
	int i_WindowSize;
	GpuProcessorContext * p_Context;
	GpuKernel * p_WindowKernel;
	GpuProcessor * p_PrevProcessor;
};

}


#endif /* GPULENGTHSLIDINGWINDOWPROCESSOR_H_ */
