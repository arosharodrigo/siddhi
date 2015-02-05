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

GpuStreamProcessor::GpuStreamProcessor(std::string _sStreamId, int _iStreamIndex, GpuMetaEvent * _pMetaEvent) :
	s_StreamId(_sStreamId),
	i_StreamIndex(_iStreamIndex),
	p_MetaEvent(_pMetaEvent),
	p_ProcessorChain(NULL),
	p_ProcessorContext(NULL)
{
	char zLogFile[256];
	sprintf(zLogFile, "logs/GpuStreamProcessor_%s.log", _sStreamId.c_str());
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

bool  GpuStreamProcessor::Initialize(int _iDeviceId, int _iInputEventBufferSize)
{
	fprintf(fp_Log, "[GpuStreamProcessor] Initialize : DeviceId=%d InputEventBufferSize=%d \n", _iDeviceId, _iInputEventBufferSize);
	fflush(fp_Log);

	if(GpuCudaHelper::SelectDevice(_iDeviceId, fp_Log))
	{

		// init ByteBuffer
		// init stream meta data
		p_ProcessorContext = new GpuProcessorContext(_iDeviceId, fp_Log);
		GpuStreamEventBuffer * pInputEventBuffer = new GpuStreamEventBuffer(_iDeviceId, p_MetaEvent, fp_Log);
		pInputEventBuffer->CreateEventBuffer(_iInputEventBufferSize);
		int iBufferIndex = p_ProcessorContext->AddEventBuffer(pInputEventBuffer);

		fprintf(fp_Log, "[GpuStreamProcessor] [Initialize] Input Event Buffer added to index=%d \n", iBufferIndex);
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
				pCurrentProcessor->Configure(pPreviousProcessor, p_ProcessorContext, fp_Log);

				pPreviousProcessor = pCurrentProcessor;
				pCurrentProcessor = pCurrentProcessor->GetNext();
			}

			// initialize
			pCurrentProcessor = p_ProcessorChain;
			while(pCurrentProcessor)
			{
				pCurrentProcessor->Init(p_MetaEvent, _iInputEventBufferSize);

				pCurrentProcessor = pCurrentProcessor->GetNext();
			}
		}

		return true;
	}
	fprintf(fp_Log, "[GpuStreamProcessor] Initialization failed \n");
	fflush(fp_Log);

	return false;
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
		return p_ProcessorChain->Process(_iNumEvents);
	}

	return 0;
}

};


