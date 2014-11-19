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
#include "CudaEvent.h"
#include "Filter.h"

namespace SiddhiGpu
{

class GpuEventConsumer;

class CudaKernelBase
{
public:
	CudaKernelBase(GpuEventConsumer * _pConsumer, FILE * _fpLog);
	virtual ~CudaKernelBase();

	virtual void Initialize() = 0;

	virtual void ProcessEvents() = 0;

	virtual void AddEvent(const CudaEvent * _pEvent) = 0;
	virtual void AddAndProcessEvents(CudaEvent ** _apEvent, int _iEventCount) = 0;

	virtual void AddFilterToDevice(Filter * _pFilter) = 0;
	virtual void CopyFiltersToDevice() = 0;

	virtual float GetElapsedTimeAverage() = 0;

protected:
	GpuEventConsumer * p_EventConsumer;
	FILE * fp_Log;
};

};

#endif /* CUDAKERNELBASE_H_ */
