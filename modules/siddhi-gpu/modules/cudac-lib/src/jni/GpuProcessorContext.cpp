/*
 * GpuProcessorContext.cpp
 *
 *  Created on: Jan 24, 2015
 *      Author: prabodha
 */


#include "GpuProcessorContext.h"
#include "GpuKernelDataTypes.h"
#include "GpuMetaEvent.h"
#include "GpuCudaHelper.h"
#include <stdlib.h>

namespace SiddhiGpu
{

GpuProcessorContext::GpuProcessorContext(int _iDeviceId, FILE * _fpLog) :
	i_DeviceId(_iDeviceId),
	fp_Log(_fpLog)
{
	fprintf(fp_Log, "[GpuProcessorContext] Created with device id : %d \n", i_DeviceId);
	fflush(fp_Log);

	vec_EventBuffers.reserve(5);
}

GpuProcessorContext::~GpuProcessorContext()
{
	fprintf(fp_Log, "[GpuProcessorContext] destroy\n");
	fflush(fp_Log);

}

int GpuProcessorContext::AddEventBuffer(GpuEventBuffer * _pEventBuffer)
{
	int iIndex = vec_EventBuffers.size();
	vec_EventBuffers.push_back(_pEventBuffer);
	return iIndex;
}

GpuEventBuffer * GpuProcessorContext::GetEventBuffer(int _iIndex)
{
	if((int)vec_EventBuffers.size() > _iIndex && _iIndex >= 0)
	{
		return vec_EventBuffers[_iIndex];
	}

	return NULL;
}

}

