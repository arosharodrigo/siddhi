/*
 * CudaKernelBase.h
 *
 *  Created on: Nov 10, 2014
 *      Author: prabodha
 */

#ifndef CUDAKERNELBASE_H_
#define CUDAKERNELBASE_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "CudaCommon.h"
#include "Filter.h"

namespace SiddhiGpu
{

class GpuEventConsumer;

class CudaKernelBase
{
public:
	CudaKernelBase(GpuEventConsumer * _pConsumer, FILE * _fpLog);
	virtual ~CudaKernelBase();

	virtual bool Initialize(int _iCudaDeviceId) = 0;
	virtual void SetEventBuffer(char * _pBuffer, int _iSize) = 0;
	void SetResultsBufferPosition(int _iPos) { i_ResultsBufferPosition = _iPos; }
	void SetEventMetaBufferPosition(int _iPos) { i_EventMetaBufferPosition = _iPos; }
	void SetSizeOfEvent(int _iSize) { i_SizeOfEvent = _iSize; }
	void SetEventDataBufferPosition(int _iPos) { i_EventDataBufferPosition = _iPos; }

	virtual void ProcessEvents(int _iNumEvents) = 0;

	virtual void AddFilterToDevice(Filter * _pFilter) = 0;
	virtual void CopyFiltersToDevice() = 0;

	virtual float GetElapsedTimeAverage() = 0;

protected:
	GpuEventConsumer * p_EventConsumer;
	FILE * fp_Log;

	int i_SizeOfEvent;
	int i_ResultsBufferPosition;
	int i_EventMetaBufferPosition;
	int i_EventDataBufferPosition;
};

};

#endif /* CUDAKERNELBASE_H_ */
