/*
 * GpuIntBuffer.h
 *
 *  Created on: Jan 29, 2015
 *      Author: prabodha
 */

#ifndef GPUINTBUFFER_H_
#define GPUINTBUFFER_H_


namespace SiddhiGpu
{

class GpuIntBuffer : public GpuEventBuffer
{
public:
	GpuIntBuffer(int _iDeviceId, GpuMetaEvent * _pMetaEvent, FILE * _fpLog);
	virtual ~GpuIntBuffer();

	void SetEventBuffer(int * _pBuffer, int _iBufferSizeInBytes, int _iEventCount);
	int * CreateEventBuffer(int _iEventCount);

	int GetMaxEventCount() { return i_EventCount; }
	int * GetHostEventBuffer() { return p_HostEventBuffer; }
	int * GetDeviceEventBuffer() { return p_DeviceEventBuffer; }
	int GetEventBufferSizeInBytes() { return i_EventBufferSizeInBytes; }

	void CopyToDevice(bool _bAsync);
	void CopyToHost(bool _bAsync);
private:
	int * p_HostEventBuffer;
	int * p_UnalignedBuffer;
	int * p_DeviceEventBuffer;

	int i_EventBufferSizeInBytes;
	int i_EventCount;
};

}

#endif /* GPUINTBUFFER_H_ */
