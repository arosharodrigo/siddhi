/*
 * GpuFilterKernel.h
 *
 *  Created on: Jan 26, 2015
 *      Author: prabodha
 */

#ifndef GPUFILTERKERNEL_H_
#define GPUFILTERKERNEL_H_

#include <stdio.h>
#include "GpuKernel.h"
#include "GpuKernelDataTypes.h"

namespace SiddhiGpu
{

class GpuMetaEvent;
class GpuProcessor;
class GpuProcessorContext;
class GpuIntBuffer;
class GpuStreamEventBuffer;

class GpuFilterKernel : public GpuKernel
{
public:
	GpuFilterKernel(GpuProcessor * _pProc, GpuProcessorContext * _pContext, int _iThreadBlockSize, FILE * _fPLog);
	~GpuFilterKernel();

	bool Initialize(GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize);
	void Process(int _iNumEvents, bool _bLast);
	char * GetResultEventBuffer();
	int GetResultEventBufferSize();

private:
	GpuProcessorContext * p_Context;
	GpuKernelFilter * p_DeviceFilter;
	GpuStreamEventBuffer * p_InputEventBuffer;
	GpuIntBuffer * p_ResultEventBuffer;
	bool b_DeviceSet;
};

}


#endif /* GPUFILTERKERNEL_H_ */
