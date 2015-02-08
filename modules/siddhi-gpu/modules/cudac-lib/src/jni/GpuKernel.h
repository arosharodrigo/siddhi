/*
 * GpuKernel.h
 *
 *  Created on: Jan 19, 2015
 *      Author: prabodha
 */

#ifndef GPUKERNEL_H_
#define GPUKERNEL_H_

#include <stdlib.h>
#include <stdio.h>

namespace SiddhiGpu
{

class GpuProcessor;
class GpuMetaEvent;

class GpuKernel
{
public:
	GpuKernel(GpuProcessor * _pProc, int _iCudaDeviceId, int _iThreadBlockSize, FILE * _fPLog);
	virtual ~GpuKernel();

	virtual bool Initialize(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize) = 0;
	virtual void Process(int _iStreamIndex, int & _iNumEvents, bool _bLast) = 0;

	virtual char * GetResultEventBuffer() = 0;
	virtual int GetResultEventBufferSize() = 0;

	int GetDeviceId() { return i_DeviceId; }

	int GetResultEventBufferIndex() { return i_ResultEventBufferIndex; }
	void SetResultEventBufferIndex(int _iIndex) { i_ResultEventBufferIndex = _iIndex; }

	int GetInputEventBufferIndex() { return i_InputBufferIndex; }
	void SetInputEventBufferIndex(int _iIndex) { i_InputBufferIndex = _iIndex; }

protected:
	int i_DeviceId;
	int i_ThreadBlockSize;
	int i_ResultEventBufferIndex;
	int i_InputBufferIndex;

	GpuProcessor * p_Processor;
	FILE * fp_Log;
};

};


#endif /* GPUKERNEL_H_ */
