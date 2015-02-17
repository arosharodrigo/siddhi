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
	GpuWindowEventBuffer(std::string _sName, int _iDeviceId, GpuMetaEvent * _pMetaEvent, FILE * _fpLog);
	virtual ~GpuWindowEventBuffer();

	char * GetReadOnlyDeviceEventBuffer() { return p_DeviceBufferPtrs[i_ReadOnlyIndex]; }
	char * GetReadWriteDeviceEventBuffer() { return p_DeviceBufferPtrs[i_ReadWriteIndex]; }
	char * GetDeviceEventBuffer() { return p_DeviceBufferPtrs[i_ReadWriteIndex]; }

	virtual char * CreateEventBuffer(int _iEventCount);

	int GetRemainingCount() { return i_RemainingCount; }
	void Sync(int _iNumEvents, bool _bAsync);
private:

	int i_RemainingCount;

	char * p_DeviceBufferPtrs[2];
	int i_ReadOnlyIndex;
	int i_ReadWriteIndex;
};

}

#endif /* GPUWINDOWEVENTBUFFER_H_ */
