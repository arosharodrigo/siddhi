/*
 * GpuEventConsumer.h
 *
 *  Created on: Oct 23, 2014
 *      Author: prabodha
 */

#ifndef GPUEVENTCONSUMER_H_
#define GPUEVENTCONSUMER_H_

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <map>
#include <vector>
#include "Timer.h"
#include "CudaKernelBase.h"
#include "CudaFilterKernel.h"
#include "CudaSingleFilterKernel.h"

namespace SiddhiGpu
{

class CudaEvent;
class Filter;

enum KernelType
{
	SingleFilterKernel = 0,
	MultiFilterKernel,
};

class GpuEventConsumer
{
public:
	GpuEventConsumer(KernelType _eKernelType, int _iMaxBufferSize, int _iEventsPerBlock);
	virtual ~GpuEventConsumer();

	void Initialize();

	void AddFilter(Filter * _pFilter);
	void ConfigureFilters();

	void ProcessEvents(int _iNumEvents);

	void PrintAverageStats();

	int GetMaxNumberOfEvents() { return i_MaxNumOfEvents; }

	void CreateByteBuffer(int _iSize);
	void SetByteBuffer(char * _pBuffer, int _iSize);
	int GetByteBufferSize() { return i_ByteBufferSize; }
	char * GetByteBuffer() { return p_ByteBuffer; }
	void SetResultsBufferPosition(int _iPos);
	void SetEventMetaBufferPosition(int _iPos);
	void SetSizeOfEvent(int _iSize);
	void SetEventDataBufferPosition(int _iPos);

private:
	typedef std::map<int, Filter *> FiltersById;

	int i_MaxNumOfEvents;
	FiltersById map_FiltersById;

	CudaKernelBase * p_CudaKernel;

	Timer m_Timer;
	FILE * fp_Log;

	char * p_ByteBuffer;
	int i_ByteBufferSize;

	int i_SizeOfEvent;
	int i_ResultsBufferPosition;
	int i_EventMetaBufferPosition;
	int i_EventDataBufferPosition;
};

};

#endif /* GPUEVENTCONSUMER_H_ */
