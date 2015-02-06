/*
 * GpuEventBuffer.h
 *
 *  Created on: Jan 28, 2015
 *      Author: prabodha
 */

#ifndef GPUEVENTBUFFER_H_
#define GPUEVENTBUFFER_H_

#include <stdio.h>
#include "GpuKernelDataTypes.h"

namespace SiddhiGpu
{

class GpuMetaEvent;

class GpuEventBuffer
{
public:
	GpuEventBuffer(std::string _sName, int _iDeviceId, GpuMetaEvent * _pMetaEvent, FILE * _fpLog);
	virtual ~GpuEventBuffer();

	virtual void CopyToDevice(bool _bAsync) = 0;
	virtual void CopyToHost(bool _bAsync) = 0;
	virtual void ResetHostEventBuffer(int _iResetVal) = 0;
	virtual void ResetDeviceEventBuffer(int _iResetVal) = 0;

	virtual void Print() = 0;

	GpuKernelMetaEvent * GetDeviceMetaEvent() { return p_DeviceMetaEvent; }
	GpuMetaEvent * GetHostMetaEvent() { return p_HostMetaEvent; }

	int GetDeviceId() { return i_DeviceId; }

protected:
	GpuMetaEvent * p_HostMetaEvent;
	GpuKernelMetaEvent * p_DeviceMetaEvent;

	std::string s_Name;
	int i_DeviceId;
	FILE * fp_Log;
};

}


#endif /* GPUEVENTBUFFER_H_ */
