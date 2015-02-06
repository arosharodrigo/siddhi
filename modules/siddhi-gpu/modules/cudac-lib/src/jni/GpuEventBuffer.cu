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

GpuEventBuffer::GpuEventBuffer(std::string _sName, int _iDeviceId, GpuMetaEvent * _pMetaEvent, FILE * _fpLog) :
	s_Name(_sName),
	i_DeviceId(_iDeviceId),
	p_HostMetaEvent(_pMetaEvent->Clone()),
	p_DeviceMetaEvent(NULL),
	fp_Log(_fpLog)
{
//	fprintf(fp_Log, "[GpuEventBuffer] <%s> Created with device id : %d \n", _sName.c_str(), i_DeviceId);
//	fflush(fp_Log);
}

GpuEventBuffer::~GpuEventBuffer()
{
	fprintf(fp_Log, "[GpuEventBuffer] <%s> destroy\n", s_Name.c_str());
	fflush(fp_Log);

	delete p_HostMetaEvent;
	p_HostMetaEvent = NULL;
}


}

#endif
