/*
 * GpuRawByteBuffer.h
 *
 *  Created on: Feb 8, 2015
 *      Author: prabodha
 */

#ifndef GPURAWBYTEBUFFER_H_
#define GPURAWBYTEBUFFER_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>

namespace SiddhiGpu
{

class GpuRawByteBuffer
{
public:
	GpuRawByteBuffer(std::string _sName, int _iDeviceId, FILE * _fpLog);
	virtual ~GpuRawByteBuffer();

	void CopyToDevice(bool _bAsync);
	void CopyToHost(bool _bAsync);
	void ResetHostEventBuffer(int _iResetVal);
	void ResetDeviceEventBuffer(int _iResetVal);

	void Print();

	int GetDeviceId() { return i_DeviceId; }

	void SetEventBuffer(char * _pBuffer, int _iBufferSizeInBytes);
	char * CreateEventBuffer(int _iBufferSizeInBytes);

	char * GetHostEventBuffer() { return p_HostEventBuffer; }
	char * GetDeviceEventBuffer() { return p_DeviceEventBuffer; }
	int GetEventBufferSizeInBytes() { return i_EventBufferSizeInBytes; }

protected:

	std::string s_Name;
	int i_DeviceId;
	FILE * fp_Log;

	char * p_HostEventBuffer;
	char * p_UnalignedBuffer;
	char * p_DeviceEventBuffer;

	int i_EventBufferSizeInBytes;
};

}


#endif /* GPURAWBYTEBUFFER_H_ */
