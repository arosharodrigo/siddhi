#ifndef _GPU_RAW_BYTE_BUFFER_CU__
#define _GPU_RAW_BYTE_BUFFER_CU__

#include "GpuKernelDataTypes.h"
#include "GpuMetaEvent.h"
#include "GpuCudaHelper.h"
#include "GpuRawByteBuffer.h"
#include <stdlib.h>
#include <stdio.h>

namespace SiddhiGpu
{

GpuRawByteBuffer::GpuRawByteBuffer(std::string _sName, int _iDeviceId, FILE * _fpLog) :
	s_Name(_sName),
	i_DeviceId(_iDeviceId),
	fp_Log(_fpLog),
	p_HostEventBuffer(NULL),
	p_UnalignedBuffer(NULL),
	p_DeviceEventBuffer(NULL),
	i_EventBufferSizeInBytes(0)
{
	fprintf(fp_Log, "[GpuRawByteBuffer] <%s> Created with device id : %d \n", _sName.c_str(), i_DeviceId);
	fflush(fp_Log);
}

GpuRawByteBuffer::~GpuRawByteBuffer()
{
	fprintf(fp_Log, "[GpuRawByteBuffer] <%s> destroy\n", s_Name.c_str());
	fflush(fp_Log);

}

void GpuRawByteBuffer::SetEventBuffer(char * _pBuffer, int _iBufferSizeInBytes)
{
	p_HostEventBuffer = _pBuffer;
	i_EventBufferSizeInBytes = _iBufferSizeInBytes;

	fprintf(fp_Log, "[GpuRawByteBuffer] <%s> Set ByteBuffer [Ptr=%p Size=%d bytes]\n",
			s_Name.c_str(), p_HostEventBuffer, i_EventBufferSizeInBytes);
	fflush(fp_Log);
}

char * GpuRawByteBuffer::CreateEventBuffer(int _iBufferSizeInBytes)
{
	i_EventBufferSizeInBytes = _iBufferSizeInBytes;
	fprintf(fp_Log, "[GpuRawByteBuffer] <%s> Allocating ByteBuffer for %d bytes \n",
			s_Name.c_str(), (int)(sizeof(char) * i_EventBufferSizeInBytes));
	fflush(fp_Log);

	GpuCudaHelper::AllocateHostMemory(true, &p_UnalignedBuffer, &p_HostEventBuffer, i_EventBufferSizeInBytes, s_Name.c_str(), fp_Log);

	CUDA_CHECK_RETURN(cudaMalloc((void**) &p_DeviceEventBuffer, i_EventBufferSizeInBytes));

	fprintf(fp_Log, "[GpuRawByteBuffer] <%s> Host ByteBuffer [Ptr=%p Size=%d]\n", s_Name.c_str(), p_HostEventBuffer, i_EventBufferSizeInBytes);
	fprintf(fp_Log, "[GpuRawByteBuffer] <%s> Device ByteBuffer [Ptr=%p] \n", s_Name.c_str(), p_DeviceEventBuffer);
	fflush(fp_Log);

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	return p_HostEventBuffer;
}

void GpuRawByteBuffer::Print()
{
	fprintf(fp_Log, "[GpuRawByteBuffer] <%s> DeviceId=%d BufferSizeInBytes=%d \n",
			s_Name.c_str(), i_DeviceId, i_EventBufferSizeInBytes);
}

void GpuRawByteBuffer::CopyToDevice(bool _bAsync)
{
#ifdef GPU_DEBUG
	fprintf(fp_Log, "[GpuRawByteBuffer] <%s> CopyToDevice : Async=%d\n", s_Name.c_str(), _bAsync);
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

void GpuRawByteBuffer::CopyToHost(bool _bAsync)
{
#ifdef GPU_DEBUG
	fprintf(fp_Log, "[GpuRawByteBuffer] <%s> CopyToHost : Async=%d\n", s_Name.c_str(), _bAsync);
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

void GpuRawByteBuffer::ResetHostEventBuffer(int _iResetVal)
{
	fprintf(fp_Log, "[GpuRawByteBuffer] <%s> HostReset : Val=%d\n", s_Name.c_str(), _iResetVal);

	memset(p_HostEventBuffer, _iResetVal, i_EventBufferSizeInBytes);
}

void GpuRawByteBuffer::ResetDeviceEventBuffer(int _iResetVal)
{
	fprintf(fp_Log, "[GpuRawByteBuffer] <%s> DeviceReset : Val=%d\n", s_Name.c_str(), _iResetVal);

	CUDA_CHECK_RETURN(cudaMemset(p_DeviceEventBuffer, _iResetVal, i_EventBufferSizeInBytes));
}

}

#endif
