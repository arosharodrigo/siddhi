/*
 * GpuStreamProcessor.cpp
 *
 *  Created on: Jan 20, 2015
 *      Author: prabodha
 */

#include "GpuMetaEvent.h"
#include "GpuProcessor.h"
#include "GpuProcessorContext.h"
#include "GpuCudaHelper.h"
#include "GpuStreamEventBuffer.h"
#include "GpuStreamProcessor.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

namespace SiddhiGpu
{

GpuStreamProcessor::GpuStreamProcessor(std::string _sQueryName, std::string _sStreamId, int _iStreamIndex, GpuMetaEvent * _pMetaEvent) :
	s_QueryName(_sQueryName),
	s_StreamId(_sStreamId),
	i_StreamIndex(_iStreamIndex),
	p_MetaEvent(_pMetaEvent->Clone()),
	p_ProcessorChain(NULL),
	p_ProcessorContext(NULL)
{
	char zLogFile[256];
	sprintf(zLogFile, "logs/GpuStreamProcessor_%s_%s.log", _sQueryName.c_str(), _sStreamId.c_str());
	fp_Log = fopen(zLogFile, "w");

	fprintf(fp_Log, "[GpuStreamProcessor] Created : Id=%s Index=%d \n", s_StreamId.c_str(), i_StreamIndex);
	fflush(fp_Log);
}

GpuStreamProcessor::~GpuStreamProcessor()
{
	if(p_ProcessorContext)
	{
		delete p_ProcessorContext;
		p_ProcessorContext = NULL;
	}

	fflush(fp_Log);
	fclose(fp_Log);
	fp_Log = NULL;
}

bool  GpuStreamProcessor::Configure(int _iDeviceId, int _iInputEventBufferSize)
{
	fprintf(fp_Log, "[GpuStreamProcessor] Configure : DeviceId=%d InputEventBufferSize=%d \n", _iDeviceId, _iInputEventBufferSize);
	fflush(fp_Log);

	if(GpuCudaHelper::SelectDevice(_iDeviceId, "GpuStreamProcessor::Configure", fp_Log))
	{

		// init ByteBuffer
		// init stream meta data
		p_ProcessorContext = new GpuProcessorContext(_iDeviceId, fp_Log);
		GpuStreamEventBuffer * pInputEventBuffer = new GpuStreamEventBuffer("MainInputEventBuffer", _iDeviceId, p_MetaEvent, fp_Log);
		pInputEventBuffer->CreateEventBuffer(_iInputEventBufferSize);
		int iBufferIndex = p_ProcessorContext->AddEventBuffer(pInputEventBuffer);

		fprintf(fp_Log, "[GpuStreamProcessor] [Configure] Input Event Buffer added to index=%d \n", iBufferIndex);
		fflush(fp_Log);
		pInputEventBuffer->Print();

		// init & configure processor chain
		if(p_ProcessorChain)
		{
			// configure
			GpuProcessor * pCurrentProcessor = p_ProcessorChain;
			GpuProcessor * pPreviousProcessor = NULL;
			while(pCurrentProcessor)
			{
				pCurrentProcessor->Configure(i_StreamIndex, pPreviousProcessor, p_ProcessorContext, fp_Log);

				pPreviousProcessor = pCurrentProcessor;
				pCurrentProcessor = pCurrentProcessor->GetNext();
			}

		}

		return true;
	}
	fprintf(fp_Log, "[GpuStreamProcessor] Configuring failed \n");
	fflush(fp_Log);

	return false;
}

void GpuStreamProcessor::Initialize(int _iInputEventBufferSize)
{
	fprintf(fp_Log, "[GpuStreamProcessor] Initialize \n");
	fflush(fp_Log);

	if(p_ProcessorChain)
	{
		// configure
		GpuProcessor * pCurrentProcessor = p_ProcessorChain;
		while(pCurrentProcessor)
		{
			pCurrentProcessor->Init(i_StreamIndex, p_MetaEvent, _iInputEventBufferSize);

			pCurrentProcessor = pCurrentProcessor->GetNext();
		}
	}
}

void GpuStreamProcessor::AddProcessor(GpuProcessor * _pProcessor)
{
	fprintf(fp_Log, "[GpuStreamProcessor] AddProcessor : Processor=%d \n", _pProcessor->GetType());
	_pProcessor->Print(fp_Log);
	fflush(fp_Log);

	if(p_ProcessorChain)
	{
		p_ProcessorChain->AddToLast(_pProcessor);
	}
	else
	{
		p_ProcessorChain = _pProcessor;
	}
}

int GpuStreamProcessor::Process(int _iNumEvents)
{
	if(p_ProcessorChain)
	{
		return p_ProcessorChain->Process(i_StreamIndex, _iNumEvents);
	}

	return 0;
}

};


