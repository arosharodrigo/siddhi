/*
 * GpuStreamEventBuffer.h
 *
 *  Created on: Jan 29, 2015
 *      Author: prabodha
 */

#ifndef GPUSTREAMEVENTBUFFER_H_
#define GPUSTREAMEVENTBUFFER_H_

#include "GpuEventBuffer.h"

namespace SiddhiGpu
{

class GpuMetaEvent;

class GpuStreamEventBuffer : public GpuEventBuffer
{
public:
	GpuStreamEventBuffer(int _iDeviceId, GpuMetaEvent * _pMetaEvent, FILE * _fpLog);
	virtual ~GpuStreamEventBuffer();

	void SetEventBuffer(char * _pBuffer, int _iBufferSizeInBytes, int _iEventCount);
	char * CreateEventBuffer(int _iEventCount);

	int GetMaxEventCount() { return i_EventCount; }
	char * GetHostEventBuffer() { return p_HostEventBuffer; }
	char * GetDeviceEventBuffer() { return p_DeviceEventBuffer; }
	int GetEventBufferSizeInBytes() { return i_EventBufferSizeInBytes; }

	void CopyToDevice(bool _bAsync);
	void CopyToHost(bool _bAsync);
	void ResetHostEventBuffer(int _iResetVal);
	void ResetDeviceEventBuffer(int _iResetVal);
	void Print();

private:
	char * p_HostEventBuffer;
	char * p_UnalignedBuffer;
	char * p_DeviceEventBuffer;

	int i_EventBufferSizeInBytes;
	int i_EventCount;
};

}


#endif /* GPUSTREAMEVENTBUFFER_H_ */
