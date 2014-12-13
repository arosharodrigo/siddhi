/*
 * GpuEventConsumer.cpp
 *
 *  Created on: Oct 23, 2014
 *      Author: prabodha
 */

#include "GpuEventConsumer.h"
#include "ByteBufferStructs.h"
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
	//PrintByteBuffer(_iNumEvents);
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

void GpuEventConsumer::PrintByteBuffer(int _iNumEvents)
{
	EventMeta * pEventMeta = (EventMeta*) (p_ByteBuffer + i_EventMetaBufferPosition);

	char * pEventDataStart = (p_ByteBuffer + i_EventDataBufferPosition);

	fprintf(fp_Log, "[PrintByteBuffer] EventMeta %d [", pEventMeta->i_AttributeCount);
	for(int i=0; i<pEventMeta->i_AttributeCount; ++i)
	{
		fprintf(fp_Log, "Pos=%d,Type=%d,Len=%d|",
				pEventMeta->a_Attributes[i].i_Position,
				pEventMeta->a_Attributes[i].i_Type,
				pEventMeta->a_Attributes[i].i_Length);
	}
	fprintf(fp_Log, "]\n");


	fprintf(fp_Log, "[PrintByteBuffer] Events Count : %d\n", _iNumEvents);
	for(int e=0; e<_iNumEvents; ++e)
	{
		char * pEvent = pEventDataStart + (i_SizeOfEvent * e);

		fprintf(fp_Log, "[PrintByteBuffer] Event_%d <%p> ", e, pEvent);

		for(int a=0; a<pEventMeta->i_AttributeCount; ++a)
		{
			switch(pEventMeta->a_Attributes[a].i_Type)
			{
				case DataType::Boolean:
				{
					int16_t i;
					memcpy(&i, pEvent + pEventMeta->a_Attributes[a].i_Position, 2);
					fprintf(fp_Log, "[Bool|Pos=%d|Len=2|Val=%d] ", pEventMeta->a_Attributes[a].i_Position, i);
				}
				break;
				case DataType::Int:
				{
					int32_t i;
					memcpy(&i, pEvent + pEventMeta->a_Attributes[a].i_Position, 4);
					fprintf(fp_Log, "[Int|Pos=%d|Len=4|Val=%d] ", pEventMeta->a_Attributes[a].i_Position, i);
				}
				break;
				case DataType::Long:
				{
					int64_t i;
					memcpy(&i, pEvent + pEventMeta->a_Attributes[a].i_Position, 8);
					fprintf(fp_Log, "[Long|Pos=%d|Len=8|Val=%lld] ", pEventMeta->a_Attributes[a].i_Position, i);
				}
				break;
				case DataType::Float:
				{
					float f;
					memcpy(&f, pEvent + pEventMeta->a_Attributes[a].i_Position, 4);
					fprintf(fp_Log, "[Float|Pos=%d|Len=4|Val=%f] ", pEventMeta->a_Attributes[a].i_Position, f);
				}
				break;
				case DataType::Double:
				{
					double f;
					memcpy(&f, pEvent + pEventMeta->a_Attributes[a].i_Position, 8);
					fprintf(fp_Log, "[Double|Pos=%d|Len=8|Val=%f] ", pEventMeta->a_Attributes[a].i_Position, f);
				}
				break;
				case DataType::StringIn:
				{
					int16_t i;
					memcpy(&i, pEvent + pEventMeta->a_Attributes[a].i_Position, 2);
					char * z = pEvent + pEventMeta->a_Attributes[a].i_Position + 2;
					z[i] = 0;
					fprintf(fp_Log, "[String|Pos=%d|Len=%d|Val=%s] ", pEventMeta->a_Attributes[a].i_Position, i, z);
				}
				break;
				default:
					break;
			}
		}

		fprintf(fp_Log, "\n");
	}
}

};
