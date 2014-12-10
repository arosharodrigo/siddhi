/*
 * GpuEventConsumer.cpp
 *
 *  Created on: Oct 23, 2014
 *      Author: prabodha
 */

#include "GpuEventConsumer.h"
#include <stdlib.h>
#include <vector>

namespace SiddhiGpu
{


GpuEventConsumer::GpuEventConsumer(KernelType _eKernelType, int _iMaxBufferSize, int _iEventsPerBlock) :
	i_MaxNumOfEvents(_iMaxBufferSize),
	i_ByteBufferSize(0),
	i_SizeOfEvent(0),
	i_ResultsBufferPosition(0),
	i_EventMetaBufferPosition(0),
	i_EventDataBufferPosition(0)
{
	fp_Log = fopen("logs/GpuEventConsumer.log", "w");

	p_ByteBuffer = NULL;

	switch(_eKernelType)
	{
		case SingleFilterKernel:
		{
			fprintf(fp_Log, "EventConsumerGpu created for SingleFilterKernel\n");
			p_CudaKernel = new CudaSingleFilterKernel(i_MaxNumOfEvents, _iEventsPerBlock, this, fp_Log);
		}
		break;
		default:
			p_CudaKernel = NULL;
			break;
	}


	fprintf(fp_Log, "EventConsumer : MaxBufferSize=[%d events]\n", i_MaxNumOfEvents);
	fflush(fp_Log);
}

GpuEventConsumer::~GpuEventConsumer()
{
	delete p_CudaKernel;

	fflush(fp_Log);
	fclose(fp_Log);
}


void GpuEventConsumer::Initialize()
{
	p_CudaKernel->Initialize();
}

void GpuEventConsumer::CreateByteBuffer(int _iSize)
{
	p_ByteBuffer = new char[_iSize];
	i_ByteBufferSize = _iSize;

	p_CudaKernel->SetEventBuffer(p_ByteBuffer, i_ByteBufferSize);

	fprintf(fp_Log, "EventConsumer : ByteBuffer Created=[%d]\n", i_ByteBufferSize);
	fflush(fp_Log);
}

void GpuEventConsumer::SetByteBuffer(char * _pBuffer, int _iSize)
{
	p_ByteBuffer = _pBuffer;
	i_ByteBufferSize = _iSize;

	p_CudaKernel->SetEventBuffer(p_ByteBuffer, i_ByteBufferSize);

	fprintf(fp_Log, "EventConsumer : ByteBuffer Set=[%d]\n", i_ByteBufferSize);
	fflush(fp_Log);
}

void GpuEventConsumer::SetResultsBufferPosition(int _iPos)
{
	i_ResultsBufferPosition = _iPos;
	p_CudaKernel->SetResultsBufferPosition(_iPos);
}

void GpuEventConsumer::SetEventMetaBufferPosition(int _iPos)
{
	i_EventMetaBufferPosition = _iPos;
	p_CudaKernel->SetEventMetaBufferPosition(_iPos);
}

void GpuEventConsumer::SetSizeOfEvent(int _iSize)
{
	i_SizeOfEvent = _iSize;
	p_CudaKernel->SetSizeOfEvent(_iSize);
}

void GpuEventConsumer::SetEventDataBufferPosition(int _iPos)
{
	i_EventDataBufferPosition = _iPos;
	p_CudaKernel->SetEventDataBufferPosition(_iPos);
}

void GpuEventConsumer::ProcessEvents(int _iNumEvents)
{
	// events are filled in bytebuffer
	// copy them to GPU
	fprintf(fp_Log, "ProcessEvents : NumEvents=%d\n", _iNumEvents);
	p_CudaKernel->ProcessEvents(_iNumEvents);
}

void GpuEventConsumer::AddFilter(Filter * _pFilter)
{
	_pFilter->Print(fp_Log);

	FiltersById::iterator ite = map_FiltersById.find(_pFilter->i_FilterId);
	if(ite == map_FiltersById.end())
	{
		map_FiltersById.insert(std::make_pair(_pFilter->i_FilterId, _pFilter));
	}
}

void GpuEventConsumer::ConfigureFilters()
{
	fprintf(fp_Log, "ConfigureFilters : FilterCount=%d\n", (int)map_FiltersById.size());

	FiltersById::iterator ite = map_FiltersById.begin();
	while(ite != map_FiltersById.end())
	{
		Filter * pFilter = ite->second;
		p_CudaKernel->AddFilterToDevice(pFilter);

		++ite;
	}
	p_CudaKernel->CopyFiltersToDevice();
	fflush(fp_Log);
}

void GpuEventConsumer::PrintAverageStats()
{
	float f = p_CudaKernel->GetElapsedTimeAverage();
	float fpe = f / i_MaxNumOfEvents;
//	printf("Average Elapsed Time (Event Batch Size : %d - %f ms) : %f ms per event\n", i_MaxBufferSize, f, fpe);
	fprintf(fp_Log, "GPU Average Elapsed Time (Event Batch Size : %d - %f ms) : %f ms per event\n", i_MaxNumOfEvents, f, fpe);
	fflush(fp_Log);
}


};
