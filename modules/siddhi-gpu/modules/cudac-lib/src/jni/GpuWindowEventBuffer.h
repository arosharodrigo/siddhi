/*
 * GpuWindowEventBuffer.h
 *
 *  Created on: Feb 15, 2015
 *      Author: prabodha
 */

#ifndef GPUWINDOWEVENTBUFFER_H_
#define GPUWINDOWEVENTBUFFER_H_

#include "GpuStreamEventBuffer.h"

namespace SiddhiGpu
{

class GpuWindowEventBuffer : public GpuStreamEventBuffer
{
public:
	GpuWindowEventBuffer();
	virtual ~GpuWindowEventBuffer();

	char * GetReadOnlyDeviceEventBuffer() { return p_ReadOnlyDeviceEventBufferPtr; }
	char * GetReadWriteDeviceEventBuffer() { return p_DeviceEventBuffer; }

	int GetRemainingCount() { return i_RemainingCount; }
private:
	void Sync();

	int i_RemainingCount;
	char * p_ReadOnlyDeviceEventBufferPtr;
};

}

#endif /* GPUWINDOWEVENTBUFFER_H_ */
