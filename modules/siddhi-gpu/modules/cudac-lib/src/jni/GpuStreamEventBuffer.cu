/*
 * GpuStreamEventBuffer.cpp
 *
 *  Created on: Jan 29, 2015
 *      Author: prabodha
 */

#ifndef _GPU_STREAM_EVENT_BUFFER_CU__
#define _GPU_STREAM_EVENT_BUFFER_CU__


#include "GpuKernelDataTypes.h"
#include "GpuMetaEvent.h"
#include "GpuCudaHelper.h"
#include "GpuStreamEventBuffer.h"
#include <stdlib.h>
#include <stdio.h>

namespace SiddhiGpu
{

GpuStreamEventBuffer::GpuStreamEventBuffer(std::string _sName, int _iDeviceId, GpuMetaEvent * _pMetaEvent, FILE * _fpLog) :
	GpuEventBuffer(_sName, _iDeviceId, _pMetaEvent, _fpLog),
	p_HostEventBuffer(NULL),
	p_UnalignedBuffer(NULL),
	p_DeviceEventBuffer(NULL),
	i_EventBufferSizeInBytes(0),
	i_EventCount(0)
{
	fprintf(fp_Log, "[GpuStreamEventBuffer] <%s> Created with device id : %d \n", _sName.c_str(), i_DeviceId);
	fflush(fp_Log);
}

GpuStreamEventBuffer::~GpuStreamEventBuffer()
{
	fprintf(fp_Log, "[GpuStreamEventBuffer] <%s> destroy\n", s_Name.c_str());
	fflush(fp_Log);

	GpuCudaHelper::FreeHostMemory(true, &p_UnalignedBuffer, &p_HostEventBuffer, i_EventBufferSizeInBytes, s_Name.c_str(), fp_Log);

//	if(p_UnalignedBuffer)
//	{
//		CUDA_CHECK_RETURN(cudaHostUnregister(p_HostEventBuffer));
//		free(p_UnalignedBuffer);
//		p_UnalignedBuffer = NULL;
//	}

	if(p_DeviceMetaEvent)
	{
		CUDA_CHECK_RETURN(cudaFree(p_DeviceMetaEvent));
	}
}

void GpuStreamEventBuffer::SetEventBuffer(char * _pBuffer, int _iBufferSizeInBytes, int _iEventCount)
{
	p_HostEventBuffer = _pBuffer;
	i_EventBufferSizeInBytes = _iBufferSizeInBytes;
	i_EventCount = _iEventCount;

	fprintf(fp_Log, "[GpuStreamEventBuffer] <%s> Set ByteBuffer [Ptr=%p Count=%d Size=%d bytes]\n",
			s_Name.c_str(), p_HostEventBuffer, i_EventCount, i_EventBufferSizeInBytes);
	fflush(fp_Log);
}

char * GpuStreamEventBuffer::CreateEventBuffer(int _iEventCount)
{
	i_EventCount = _iEventCount;
	i_EventBufferSizeInBytes = _iEventCount * p_HostMetaEvent->i_SizeOfEventInBytes;
	fprintf(fp_Log, "[GpuStreamEventBuffer] <%s> Allocating ByteBuffer for %d events : %d bytes \n",
			s_Name.c_str(), _iEventCount, (int)(sizeof(char) * i_EventBufferSizeInBytes));
	fflush(fp_Log);

	GpuCudaHelper::AllocateHostMemory(true, &p_UnalignedBuffer, &p_HostEventBuffer, i_EventBufferSizeInBytes, s_Name.c_str(), fp_Log);

	CUDA_CHECK_RETURN(cudaMalloc((void**) &p_DeviceEventBuffer, i_EventBufferSizeInBytes));

	fprintf(fp_Log, "[GpuStreamEventBuffer] <%s> Host ByteBuffer [Ptr=%p Size=%d]\n", s_Name.c_str(), p_HostEventBuffer, i_EventBufferSizeInBytes);
	fprintf(fp_Log, "[GpuStreamEventBuffer] <%s> Device ByteBuffer [Ptr=%p] \n", s_Name.c_str(), p_DeviceEventBuffer);
	fflush(fp_Log);

	int GpuMetaEventSize = sizeof(GpuKernelMetaEvent) + sizeof(GpuKernelMetaAttribute) * p_HostMetaEvent->i_AttributeCount;

	CUDA_CHECK_RETURN(cudaMalloc((void**) &p_DeviceMetaEvent, GpuMetaEventSize));

	GpuKernelMetaEvent * pHostMetaEvent = (GpuKernelMetaEvent*) malloc(GpuMetaEventSize);

	pHostMetaEvent->i_StreamIndex = p_HostMetaEvent->i_StreamIndex;
	pHostMetaEvent->i_AttributeCount = p_HostMetaEvent->i_AttributeCount;
	pHostMetaEvent->i_SizeOfEventInBytes = p_HostMetaEvent->i_SizeOfEventInBytes;

	for(int i=0; i<p_HostMetaEvent->i_AttributeCount; ++i)
	{
		pHostMetaEvent->p_Attributes[i].i_Type = p_HostMetaEvent->p_Attributes[i].i_Type;
		pHostMetaEvent->p_Attributes[i].i_Position = p_HostMetaEvent->p_Attributes[i].i_Position;
		pHostMetaEvent->p_Attributes[i].i_Length = p_HostMetaEvent->p_Attributes[i].i_Length;
	}

	CUDA_CHECK_RETURN(cudaMemcpy(
		p_DeviceMetaEvent,
		pHostMetaEvent,
		GpuMetaEventSize,
		cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	free(pHostMetaEvent);
	pHostMetaEvent = NULL;

	return p_HostEventBuffer;
}

void GpuStreamEventBuffer::Print()
{
	fprintf(fp_Log, "[GpuStreamEventBuffer] <%s> DeviceId=%d MaxEventCount=%d BufferSizeInBytes=%d \n",
			s_Name.c_str(), i_DeviceId, i_EventCount, i_EventBufferSizeInBytes);
	fprintf(fp_Log, "[GpuStreamEventBuffer] <%s> EventMeta %d [", s_Name.c_str(), p_HostMetaEvent->i_AttributeCount);
	for(int i=0; i<p_HostMetaEvent->i_AttributeCount; ++i)
	{
		fprintf(fp_Log, "Pos=%d,Type=%d,Len=%d|",
				p_HostMetaEvent->p_Attributes[i].i_Position,
				p_HostMetaEvent->p_Attributes[i].i_Type,
				p_HostMetaEvent->p_Attributes[i].i_Length);
	}
	fprintf(fp_Log, "]\n");
}

void GpuStreamEventBuffer::CopyToDevice(bool _bAsync)
{
#ifdef GPU_DEBUG
	fprintf(fp_Log, "[GpuStreamEventBuffer] <%s> CopyToDevice : Async=%d\n", s_Name.c_str(), _bAsync);
#endif

	if(_bAsync)
	{
		CUDA_CHECK_RETURN(cudaMemcpyAsync(p_DeviceEventBuffer, p_HostEventBuffer, i_EventBufferSizeInBytes, cudaMemcpyHostToDevice));
	}
	else
	{
		CUDA_CHECK_RETURN(cudaMemcpy(p_DeviceEventBuffer, p_HostEventBuffer, i_EventBufferSizeInBytes, cudaMemcpyHostToDevice));
	}
}

void GpuStreamEventBuffer::CopyToHost(bool _bAsync)
{
#ifdef GPU_DEBUG
	fprintf(fp_Log, "[GpuStreamEventBuffer] <%s> CopyToHost : Async=%d\n", s_Name.c_str(), _bAsync);
#endif

	if(_bAsync)
	{
		CUDA_CHECK_RETURN(cudaMemcpyAsync(p_HostEventBuffer, p_DeviceEventBuffer, i_EventBufferSizeInBytes, cudaMemcpyDeviceToHost));
	}
	else
	{
		CUDA_CHECK_RETURN(cudaMemcpy(p_HostEventBuffer, p_DeviceEventBuffer, i_EventBufferSizeInBytes, cudaMemcpyDeviceToHost));
	}
}

void GpuStreamEventBuffer::ResetHostEventBuffer(int _iResetVal)
{
	fprintf(fp_Log, "[GpuStreamEventBuffer] <%s> HostReset : Val=%d\n", s_Name.c_str(), _iResetVal);

	memset(p_HostEventBuffer, _iResetVal, i_EventBufferSizeInBytes);
}

void GpuStreamEventBuffer::ResetDeviceEventBuffer(int _iResetVal)
{
	fprintf(fp_Log, "[GpuStreamEventBuffer] <%s> DeviceReset : Val=%d\n", s_Name.c_str(), _iResetVal);

	CUDA_CHECK_RETURN(cudaMemset(p_DeviceEventBuffer, _iResetVal, i_EventBufferSizeInBytes));
}

}

#endif

