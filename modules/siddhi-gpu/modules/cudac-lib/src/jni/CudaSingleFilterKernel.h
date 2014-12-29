/*
 * CudaSingleFilterKernel.h
 *
 *  Created on: Nov 9, 2014
 *      Author: prabodha
 */

#ifndef CUDASINGLEFILTERKERNEL_H_
#define CUDASINGLEFILTERKERNEL_H_

#include "CudaKernelBase.h"
#include <list>
#include "helper_timer.h"

namespace SiddhiGpu
{

// -------------------------------------------------------------------------------------------------------------------

typedef struct SingleFilterKernelInput
{
	char * 					  p_ByteBuffer;         // ByteBuffer from java side
	int                       i_ResultsPosition;    // Results array position in ByteBuffer
	int                       i_EventMetaPosition;  // EventMeta position in ByteBuffer
	int                       i_EventDataPosition;  // EventData position in ByteBuffer
	int                       i_SizeOfEvent;        // Size of an event
	int                       i_EventsPerBlock;     // number of events allocated per block
	Filter *                  ap_Filter;            // Filters buffer - pre-copied at initialization
	int                       i_MaxEventCount;      // used for setting results array
	int                       i_EventCount;         // Num events in this batch
} SingleFilterKernelInput;

// -------------------------------------------------------------------------------------------------------------------

class CudaSingleFilterKernel : public CudaKernelBase
{
public:
	CudaSingleFilterKernel(int _iMaxBufferSize, GpuEventConsumer * _pConsumer, FILE * _fpLog);
	CudaSingleFilterKernel(int _iMaxBufferSize, int _iEventsPerBlock, GpuEventConsumer * _pConsumer, FILE * _fpLog);
	virtual ~CudaSingleFilterKernel();

	bool Initialize(int _iCudaDeviceId);
	void SetEventBuffer(char * _pBuffer, int _iSize);
	char * GetEventBuffer(int _iSize);
	void ProcessEvents(int _iNumEvents);

	void AddFilterToDevice(Filter * _pFilter);
	void CopyFiltersToDevice();

	float GetElapsedTimeAverage();

private:
	bool SelectDevice(int _iDeviceId);

	int i_CudaDeviceId;
	bool b_DeviceSet;

	int i_EventsPerBlock;
	int i_MaxNumberOfEvents;
//	int i_NumEvents;
	int i_NumAttributes;

	char * p_HostEventBuffer;
	char * p_UnalignedBuffer;
	int i_EventBufferSize;
//	CudaEvent * ap_HostEventBuffer;
	std::list<Filter*> lst_HostFilters;

//	int * pi_HostMachedEvents;
//	int * pi_DeviceMatchedEvents;

	::StopWatchInterface * p_StopWatch;

	std::list<float> lst_ElapsedTimes;

	SingleFilterKernelInput * p_DeviceInput;
	SingleFilterKernelInput * p_HostInput;
};

};

#endif /* CUDASINGLEFILTERKERNEL_H_ */
