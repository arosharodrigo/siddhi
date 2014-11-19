/*
 * CudaFilterKernel.h
 *
 *  Created on: Oct 18, 2014
 *      Author: prabodha
 */

#ifndef CUDAFILTERKERNEL_H_
#define CUDAFILTERKERNEL_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <list>
#include "CudaEvent.h"
#include "Filter.h"
#include "CudaKernelBase.h"

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

class StopWatchInterface;

namespace SiddhiGpu
{

// -------------------------------------------------------------------------------------------------------------------

typedef struct MultipleFilterKernelInput
{
	CudaEvent *               ap_EventBuffer;       // input events in device memory
	int                       i_EventCount;         // number of events in buffer
	Filter *                  ap_Filters;           // Filters buffer - pre-copied at initialization
	int                       i_FilterCount;        // number of filters in buffer
	int                       i_MaxEventCount;      // used for setting results array
} MultipleFilterKernelInput;

// -------------------------------------------------------------------------------------------------------------------

class GpuEventConsumer;

class CudaFilterKernel : public CudaKernelBase
{
public:
	CudaFilterKernel(int _iMaxBufferSize, GpuEventConsumer * _pConsumer, FILE * _fpLog);
	virtual ~CudaFilterKernel();

	virtual void Initialize();

	virtual void ProcessEvents();

	virtual void AddEvent(const CudaEvent * _pEvent);
	virtual void AddAndProcessEvents(CudaEvent ** _apEvent, int _iEventCount);

	virtual void AddFilterToDevice(Filter * _pFilter);
	virtual void CopyFiltersToDevice();

	float GetElapsedTimeAverage();

	static void OnExit();

private:
	int i_MaxEventBufferSize;
	int i_NumEvents;
	int i_NumAttributes;

	CudaEvent * ap_HostEventBuffer;
	std::list<Filter*> lst_HostFilters;
	int i_FilterCount;

	int * pi_HostMachedEvents;
	int * pi_DeviceMatchedEvents;

	::StopWatchInterface * p_StopWatch;

	std::list<float> lst_ElapsedTimes;

	MultipleFilterKernelInput * p_DeviceInput;
	MultipleFilterKernelInput * p_HostInput;
};

};

#endif /* CUDAFILTERKERNEL_H_ */
