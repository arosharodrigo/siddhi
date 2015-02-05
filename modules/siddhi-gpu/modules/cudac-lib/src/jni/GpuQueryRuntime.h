/*
 * GpuQueryRuntime.h
 *
 *  Created on: Jan 18, 2015
 *      Author: prabodha
 */

#include <stdlib.h>
#include <stdio.h>
#include <vector>

#ifndef GPUQUERYRUNTIME_H_
#define GPUQUERYRUNTIME_H_

#include <stdlib.h>
#include <stdio.h>
#include <map>
#include <vector>
#include <string.h>
#include <string>

#include "GpuMetaEvent.h"

namespace SiddhiGpu
{

class GpuStreamProcessor;
class GpuProcessor;

class GpuQueryRuntime
{
public:
	typedef std::vector<GpuStreamProcessor*> StreamProcessors;
	typedef std::map<std::string, GpuStreamProcessor*> StreamProcessorsByStreamId;

	GpuQueryRuntime(std::string _zQueryName, int _iDeviceId, int _iInputEventBufferSize);
	~GpuQueryRuntime();

	void AddStream(std::string _sStramId, GpuMetaEvent * _pMetaEvent);
	GpuStreamProcessor * GetStream(std::string _sStramId);
	void AddProcessor(std::string _sStramId, GpuProcessor * _pProcessor);

	char * GetInputEventBuffer(std::string _sStramId);
	int GetInputEventBufferSizeInBytes(std::string _sStramId);

	bool Configure();

private:
	std::string s_QueryName;
	int i_DeviceId;
	int i_InputEventBufferSize;
	StreamProcessors vec_StreamProcessors;
	StreamProcessorsByStreamId map_StreamProcessorsByStreamId;
	FILE * fp_Log;
};


};


#endif /* GPUQUERYRUNTIME_H_ */
