/*
 * GpuIntBuffer.h
 *
 *  Created on: Jan 29, 2015
 *      Author: prabodha
 */

#ifndef GPUINTBUFFER_H_
#define GPUINTBUFFER_H_

#include <stdio.h>
#include "GpuEventBuffer.h"

namespace SiddhiGpu
{

class GpuMetaEvent;

class GpuIntBuffer : public GpuEventBuffer
{
public:
	GpuIntBuffer(std::string _sName, int _iDeviceId, GpuMetaEvent * _pMetaEvent, FILE * _fpLog);
	virtual ~GpuIntBuffer();

	void SetEventBuffer(int * _pBuffer, int _iBufferSizeInBytes, int _iEventCount);
	int * CreateEventBuffer(int _iEventCount);

	int GetMaxEventCount() { return i_EventCount; }
	int * GetHostEventBuffer() { return p_HostEventBuffer; }
	int * GetDeviceEventBuffer() { return p_DeviceEventBuffer; }
	unsigned int GetEventBufferSizeInBytes() { return i_EventBufferSizeInBytes; }

	void CopyToDevice(bool _bAsync);
	void CopyToHost(bool _bAsync);
	void ResetHostEventBuffer(int _iResetVal);
	void ResetDeviceEventBuffer(int _iResetVal);
	void Print();

private:
	int * p_HostEventBuffer;
	int * p_UnalignedBuffer;
	int * p_DeviceEventBuffer;

	unsigned int i_EventBufferSizeInBytes;
	int i_EventCount;
};

}

#endif /* GPUINTBUFFER_H_ */
