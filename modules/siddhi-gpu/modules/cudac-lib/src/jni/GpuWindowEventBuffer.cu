#ifndef __GPU_WINDOW_EVENT_BUFFER_CU_
#define __GPU_WINDOW_EVENT_BUFFER_CU_

#include "GpuWindowEventBuffer.h"

namespace SiddhiGpu
{

GpuWindowEventBuffer::GpuWindowEventBuffer(std::string _sName, int _iDeviceId, GpuMetaEvent * _pMetaEvent, FILE * _fpLog) :
		GpuStreamEventBuffer(_sName, _iDeviceId, _pMetaEvent, _fpLog),
		i_RemainingCount(0),
		p_ReadOnlyDeviceEventBufferPtr(NULL)
{
	fprintf(fp_Log, "[GpuWindowEventBuffer] <%s> Created with device id : %d \n", _sName.c_str(), i_DeviceId);
	fflush(fp_Log);
}

GpuWindowEventBuffer::~GpuWindowEventBuffer()
{
	fprintf(fp_Log, "[GpuWindowEventBuffer] <%s> destroy\n", s_Name.c_str());
	fflush(fp_Log);


	if(p_ReadOnlyDeviceEventBufferPtr)
	{
		CUDA_CHECK_RETURN(cudaFree(p_ReadOnlyDeviceEventBufferPtr));
		p_ReadOnlyDeviceEventBufferPtr = NULL;
	}
}

char * GpuWindowEventBuffer::CreateEventBuffer(int _iEventCount)
{
	i_EventCount = _iEventCount;
	i_RemainingCount = i_EventCount;

	i_EventBufferSizeInBytes = _iEventCount * p_HostMetaEvent->i_SizeOfEventInBytes;
	fprintf(fp_Log, "[GpuWindowEventBuffer] <%s> Allocating ByteBuffer for %d events (x %d) : %d bytes \n",
			s_Name.c_str(), _iEventCount, p_HostMetaEvent->i_SizeOfEventInBytes, (int)(sizeof(char) * i_EventBufferSizeInBytes));
	fflush(fp_Log);

	GpuCudaHelper::AllocateHostMemory(true, &p_UnalignedBuffer, &p_HostEventBuffer, i_EventBufferSizeInBytes, s_Name.c_str(), fp_Log);

	CUDA_CHECK_RETURN(cudaMalloc((void**) &p_DeviceEventBuffer, i_EventBufferSizeInBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &p_ReadOnlyDeviceEventBufferPtr, i_EventBufferSizeInBytes));

	fprintf(fp_Log, "[GpuWindowEventBuffer] <%s> Host ByteBuffer [Ptr=%p Size=%d]\n", s_Name.c_str(), p_HostEventBuffer, i_EventBufferSizeInBytes);
	fprintf(fp_Log, "[GpuWindowEventBuffer] <%s> Device ReadWrite ByteBuffer [Ptr=%p] \n", s_Name.c_str(), p_DeviceEventBuffer);
	fprintf(fp_Log, "[GpuWindowEventBuffer] <%s> Device ReadOnly ByteBuffer [Ptr=%p] \n", s_Name.c_str(), p_ReadOnlyDeviceEventBufferPtr);
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

void GpuWindowEventBuffer::Sync(int _iNumEvents, bool _bAsync)
{
#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(fp_Log, "[GpuWindowEventBuffer] <%s> Sync : Async=%d\n", s_Name.c_str(), _bAsync);
#endif

#ifdef _GLIBCXX_ATOMIC_BUILTINS

	// swap two buffers
	// priority is for readonly buffer

	char* tmp = p_ReadOnlyDeviceEventBufferPtr;
	while(!__sync_bool_compare_and_swap(
			&p_ReadOnlyDeviceEventBufferPtr,
			p_ReadOnlyDeviceEventBufferPtr,
			p_DeviceEventBuffer)){};
	while(!__sync_bool_compare_and_swap(
			&p_DeviceEventBuffer,
			p_DeviceEventBuffer,
			tmp)){};

#else
  #error no atomic operations available!
#endif

	if(_iNumEvents > i_RemainingCount)
	{
		i_RemainingCount = 0;
	}
	else
	{
		i_RemainingCount -= _iNumEvents;
	}

	if(_bAsync)
	{
		// update readwrite copy with swapped readonly buffer
		CUDA_CHECK_RETURN(cudaMemcpyAsync(
				p_DeviceEventBuffer,
				p_ReadOnlyDeviceEventBufferPtr,
				i_EventBufferSizeInBytes,
				cudaMemcpyDeviceToDevice));
	}
	else
	{
		// update readwrite copy with swapped readonly buffer
		CUDA_CHECK_RETURN(cudaMemcpy(
				p_DeviceEventBuffer,
				p_ReadOnlyDeviceEventBufferPtr,
				i_EventBufferSizeInBytes,
				cudaMemcpyDeviceToDevice));
	}
}

}

#endif

