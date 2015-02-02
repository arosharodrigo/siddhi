#ifndef _GPU_EVENT_BUFFER_CU__
#define _GPU_EVENT_BUFFER_CU__

#include "GpuKernelDataTypes.h"
#include "GpuMetaEvent.h"
#include "GpuCudaHelper.h"
#include "GpuEventBuffer.h"
#include <stdlib.h>
#include <stdio.h>

namespace SiddhiGpu
{

GpuEventBuffer::GpuEventBuffer(int _iDeviceId, GpuMetaEvent * _pMetaEvent, FILE * _fpLog) :
	i_DeviceId(_iDeviceId),
	p_HostMetaEvent(_pMetaEvent->Clone()),
	p_DeviceMetaEvent(NULL),
	fp_Log(_fpLog)
{
	fprintf(fp_Log, "[GpuEventBuffer] Created with device id : %d \n", i_DeviceId);
	fflush(fp_Log);
}

GpuEventBuffer::~GpuEventBuffer()
{
	fprintf(fp_Log, "[GpuEventBuffer] destroy\n");
	fflush(fp_Log);

	delete p_HostMetaEvent;
	p_HostMetaEvent = NULL;
}


}

#endif
