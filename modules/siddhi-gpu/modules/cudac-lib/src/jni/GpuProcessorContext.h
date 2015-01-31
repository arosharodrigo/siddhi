/*
 * GpuProcessorContext.h
 *
 *  Created on: Jan 24, 2015
 *      Author: prabodha
 */

#ifndef GPUPROCESSORCONTEXT_H_
#define GPUPROCESSORCONTEXT_H_

#include <stdio.h>
#include <vector>
#include "GpuKernelDataTypes.h"

namespace SiddhiGpu
{

class GpuMetaEvent;
class GpuEventBuffer;

class GpuProcessorContext
{
public:
	GpuProcessorContext(int _iDeviceId, FILE * _fpLog);
	~GpuProcessorContext();

	int AddEventBuffer(GpuEventBuffer * _pEventBuffer);
	GpuEventBuffer * GetEventBuffer(int _iIndex);

	int GetDeviceId() { return i_DeviceId; }

private:
	std::vector<GpuEventBuffer*> vec_EventBuffers;

	int i_DeviceId;
	FILE * fp_Log;
};

}



#endif /* GPUPROCESSORCONTEXT_H_ */
