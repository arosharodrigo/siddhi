/*
 * CudaSingleFilterKernel.h
 *
 *  Created on: Nov 9, 2014
 *      Author: prabodha
 */

#ifndef CUDASINGLEFILTERKERNEL_H_
#define CUDASINGLEFILTERKERNEL_H_

#include "CudaKernelBase.h"

namespace SiddhiGpu
{

// -------------------------------------------------------------------------------------------------------------------

typedef struct SingleFilterKernelInput
{
	CudaEvent *               ap_EventBuffer;       // input events in device memory
	int                       i_EventCount;         // number of events in buffer
	Filter *                  ap_Filter;            // Filters buffer - pre-copied at initialization
	int                       i_EventsPerBlock;     // number of events allocated per block
	int                       i_MaxEventCount;      // used for setting results array
} SingleFilterKernelInput;

// -------------------------------------------------------------------------------------------------------------------

class CudaSingleFilterKernel : public CudaKernelBase
{
public:
	CudaSingleFilterKernel(int _iMaxBufferSize, GpuEventConsumer * _pConsumer, FILE * _fpLog);
	CudaSingleFilterKernel(int _iMaxBufferSize, int _iEventsPerBlock, GpuEventConsumer * _pConsumer, FILE * _fpLog);
	virtual ~CudaSingleFilterKernel();

	void Initialize();
	void ProcessEvents();

	void AddEvent(const CudaEvent * _pEvent);
	void AddAndProcessEvents(CudaEvent ** _apEvent, int _iEventCount);

	void AddFilterToDevice(Filter * _pFilter);
	void CopyFiltersToDevice();

	float GetElapsedTimeAverage();

private:
	int i_EventsPerBlock;
	int i_MaxEventBufferSize;
	int i_NumEvents;
	int i_NumAttributes;

	CudaEvent * ap_HostEventBuffer;
	std::list<Filter*> lst_HostFilters;

	int * pi_HostMachedEvents;
	int * pi_DeviceMatchedEvents;

	::StopWatchInterface * p_StopWatch;

	std::list<float> lst_ElapsedTimes;

	SingleFilterKernelInput * p_DeviceInput;
	SingleFilterKernelInput * p_HostInput;
};

};

#endif /* CUDASINGLEFILTERKERNEL_H_ */
