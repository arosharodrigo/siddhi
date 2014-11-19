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
	void OnEvents(CudaEvent ** _apEvents, int _iEventCount);

	void AddFilter(Filter * _pFilter);
	void ConfigureFilters();

	void OnCudaEventMatch(int _iEventPos, int _iFilterId);
	void OnEventMatch(CudaEvent * _pEvent, int _iFilterId);

	std::vector<int> GetMatchingEvents();

	void PrintAverageStats();

	int GetMaxBufferSize() { return i_MaxBufferSize; }

private:
	typedef std::map<int, Filter *> FiltersById;
	typedef std::vector<int> ResultsEvents;

	int i_MaxBufferSize;
	FiltersById map_FiltersById;

	CudaKernelBase * p_CudaKernel;

	ResultsEvents vec_Result;
	Timer m_Timer;
	FILE * fp_Log;
};

};

#endif /* GPUEVENTCONSUMER_H_ */
