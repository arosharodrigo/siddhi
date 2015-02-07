/*
 * GpuQueryRuntime.cpp
 *
 *  Created on: Jan 18, 2015
 *      Author: prabodha
 */

#include "GpuProcessor.h"
#include "GpuMetaEvent.h"
#include "GpuStreamProcessor.h"
#include "GpuProcessorContext.h"
#include "GpuUtils.h"
#include "GpuCudaHelper.h"
#include "GpuStreamEventBuffer.h"
#include "GpuQueryRuntime.h"

namespace SiddhiGpu
{

GpuQueryRuntime::GpuQueryRuntime(std::string _zQueryName, int _iDeviceId, int _iInputEventBufferSize) :
	s_QueryName(_zQueryName),
	i_DeviceId(_iDeviceId),
	i_InputEventBufferSize(_iInputEventBufferSize)
{
	char zLogFile[256];
	sprintf(zLogFile, "logs/GpuQueryRuntime_%s.log", _zQueryName.c_str());
	fp_Log = fopen(zLogFile, "w");
	if(fp_Log)
	{
		fprintf(fp_Log, "[GpuQueryRuntime] Created : Name=%s deviceId=%d InputBufferSize=%d \n",
				s_QueryName.c_str(), i_DeviceId, i_InputEventBufferSize);
		fflush(fp_Log);

		GpuUtils::PrintThreadInfo(s_QueryName.c_str(), fp_Log);
	}
	else
	{
		fp_Log = NULL;
		printf("[GpuQueryRuntime] Error while creating log file at %s\n", zLogFile);
	}
}

GpuQueryRuntime::~GpuQueryRuntime()
{
	GpuCudaHelper::DeviceReset();

	fflush(fp_Log);
	fclose(fp_Log);
}

void GpuQueryRuntime::AddStream(std::string _sStramId, GpuMetaEvent * _pMetaEvent)
{
	fprintf(fp_Log, "[GpuQueryRuntime] AddStream : Id=%s Index=%d \n", _sStramId.c_str(), _pMetaEvent->i_StreamIndex);
	fflush(fp_Log);

	GpuStreamProcessor * pStreamProcessor = new GpuStreamProcessor(s_QueryName, _sStramId, _pMetaEvent->i_StreamIndex, _pMetaEvent);
	vec_StreamProcessors.push_back(pStreamProcessor);
	map_StreamProcessorsByStreamId.insert(std::make_pair(_sStramId, pStreamProcessor));
}

void GpuQueryRuntime::AddProcessor(std::string _sStramId, GpuProcessor * _pProcessor)
{
	fprintf(fp_Log, "[GpuQueryRuntime] AddProcessor : Id=%s Type=%d \n", _sStramId.c_str(), _pProcessor->GetType());
	fflush(fp_Log);

	StreamProcessorsByStreamId::iterator ite = map_StreamProcessorsByStreamId.find(_sStramId);
	if(ite != map_StreamProcessorsByStreamId.end())
	{
		ite->second->AddProcessor(_pProcessor);
	}
}

GpuStreamProcessor * GpuQueryRuntime::GetStream(std::string _sStramId)
{
	fprintf(fp_Log, "[GpuQueryRuntime] GetStream : Id=%s \n", _sStramId.c_str());
	fflush(fp_Log);

	StreamProcessorsByStreamId::iterator ite = map_StreamProcessorsByStreamId.find(_sStramId);
	if(ite != map_StreamProcessorsByStreamId.end())
	{
		return ite->second;
	}

	return NULL;
}

char * GpuQueryRuntime::GetInputEventBuffer(std::string _sStramId)
{
	fprintf(fp_Log, "[GpuQueryRuntime] GetInputEventBuffer : Id=%s \n", _sStramId.c_str());
	fflush(fp_Log);

	StreamProcessorsByStreamId::iterator ite = map_StreamProcessorsByStreamId.find(_sStramId);
	if(ite != map_StreamProcessorsByStreamId.end())
	{
		GpuEventBuffer * pEventBuffer = ite->second->GetProcessorContext()->GetEventBuffer(0);
		if(pEventBuffer)
		{
			return ((GpuStreamEventBuffer*)pEventBuffer)->GetHostEventBuffer();
		}
	}

	return NULL;
}

int GpuQueryRuntime::GetInputEventBufferSizeInBytes(std::string _sStramId)
{
	fprintf(fp_Log, "[GpuQueryRuntime] GetInputEventBufferSizeInBytes : Id=%s \n", _sStramId.c_str());
	fflush(fp_Log);

	StreamProcessorsByStreamId::iterator ite = map_StreamProcessorsByStreamId.find(_sStramId);
	if(ite != map_StreamProcessorsByStreamId.end())
	{
		GpuEventBuffer * pEventBuffer = ite->second->GetProcessorContext()->GetEventBuffer(0);
		if(pEventBuffer)
		{
			return ((GpuStreamEventBuffer*)pEventBuffer)->GetEventBufferSizeInBytes();
		}
	}

	return 0;
}

bool GpuQueryRuntime::Configure()
{
	fprintf(fp_Log, "[GpuQueryRuntime] Configure \n");
	fflush(fp_Log);
	GpuUtils::PrintThreadInfo(s_QueryName.c_str(), fp_Log);

	// create kernels
	// arrange kernels in proper order
	// copy meta data to GPU

	StreamProcessorsByStreamId::iterator ite = map_StreamProcessorsByStreamId.begin();
	while(ite != map_StreamProcessorsByStreamId.end())
	{
		if(!ite->second->Initialize(i_DeviceId, i_InputEventBufferSize))
		{
			return false;
		}

		++ite;
	}

	return true;
}

};

