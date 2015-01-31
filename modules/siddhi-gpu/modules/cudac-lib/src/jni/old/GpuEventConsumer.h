/*
 * GpuEventConsumer.h
 *
 *  Created on: Oct 23, 2014
 *      Author: prabodha
 */

#ifndef OLD_GPUEVENTCONSUMER_H_
#define OLD_GPUEVENTCONSUMER_H_

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <map>
#include "CudaSingleFilterKernel.h"
#include "CudaKernelBase.h"
#include "Timer.h"



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
	GpuEventConsumer(KernelType _eKernelType, const char * _zName, int _iMaxBufferSize, int _iEventsPerBlock);
	virtual ~GpuEventConsumer();

	bool Initialize(int _iCudaDeviceId);

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

	void PrintByteBuffer(int _iNumEvents);
	void EvaluateEvenetsInCpu(int _iNumEvents);
	void PrintThreadInfo();

	int i_MaxNumOfEvents;
	FiltersById map_FiltersById;
	Filter * p_Filter;

	CudaKernelBase * p_CudaKernel;

	Timer m_Timer;
	FILE * fp_Log;

	char * p_ByteBuffer;
	int i_ByteBufferSize;

	int i_EventsPerBlock;
	int i_SizeOfEvent;
	int i_ResultsBufferPosition;
	int i_EventMetaBufferPosition;
	int i_EventDataBufferPosition;

	char z_Name[256];
};

};

#endif /*OLD_GPUEVENTCONSUMER_H_R_H_ */
