/*
 * GpuJoinProcessor.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: prabodha
 */

#include "GpuMetaEvent.h"
#include "GpuJoinKernel.h"
#include "GpuProcessorContext.h"
#include "GpuJoinProcessor.h"

namespace SiddhiGpu
{

GpuJoinProcessor::GpuJoinProcessor(int _iLeftWindowSize, int _iRightWindowSize) :
	GpuProcessor(GpuProcessor::LENGTH_SLIDING_WINDOW),
	i_LeftStraemWindowSize(_iLeftWindowSize),
	i_RightStraemWindowSize(_iRightWindowSize),
	p_Context(NULL),
	p_JoinKernel(NULL),
	p_PrevProcessor(NULL)
{

}

GpuJoinProcessor::~GpuJoinProcessor()
{
	if(p_JoinKernel)
	{
		delete p_JoinKernel;
		p_JoinKernel = NULL;
	}

	p_Context = NULL;
	p_PrevProcessor = NULL;
}

GpuProcessor * GpuJoinProcessor::Clone()
{
	GpuJoinProcessor * pCloned = new GpuJoinProcessor(i_LeftStraemWindowSize, i_RightStraemWindowSize);

	return pCloned;
}

void GpuJoinProcessor::Configure(GpuProcessor * _pPrevProcessor, GpuProcessorContext * _pContext, FILE * _fpLog)
{
	fp_Log = _fpLog;
	p_Context = _pContext;
	p_PrevProcessor = _pPrevProcessor;

	fprintf(fp_Log, "[GpuJoinProcessor] Configure : PrevProcessor=%p Context=%p \n", _pPrevProcessor, p_Context);
	fflush(fp_Log);

}

void GpuJoinProcessor::Init(GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize)
{
	fprintf(fp_Log, "[GpuJoinProcessor] Init \n");
	fflush(fp_Log);

	if(p_PrevProcessor)
	{

		switch(p_PrevProcessor->GetType())
		{
		case GpuProcessor::FILTER:
		{
			p_JoinKernel = new GpuJoinKernel(this, p_Context, i_ThreadBlockSize, i_LeftStraemWindowSize, i_RightStraemWindowSize, fp_Log);
			p_JoinKernel->SetInputEventBufferIndex(p_PrevProcessor->GetResultEventBufferIndex());
		}
		break;
		default:
			break;
		}
	}
	else
	{
		p_JoinKernel = new GpuJoinKernel(this, p_Context, i_ThreadBlockSize, i_LeftStraemWindowSize, i_RightStraemWindowSize, fp_Log);
		p_JoinKernel->SetInputEventBufferIndex(p_PrevProcessor->GetResultEventBufferIndex());
	}

	p_JoinKernel->Initialize(_pMetaEvent, _iInputEventBufferSize);

}

int GpuJoinProcessor::Process(int _iNumEvents)
{
	fprintf(fp_Log, "[GpuJoinProcessor] Process : NumEvents=%d \n", _iNumEvents);
	fflush(fp_Log);

	p_JoinKernel->Process(_iNumEvents, (p_Next == NULL));

	if(p_Next)
	{
		_iNumEvents = p_Next->Process(_iNumEvents);
	}

	return _iNumEvents;
}

void GpuJoinProcessor::Print(FILE * _fp)
{

}

int GpuJoinProcessor::GetResultEventBufferIndex()
{
	return p_JoinKernel->GetResultEventBufferIndex();
}

char * GpuJoinProcessor::GetResultEventBuffer()
{
	return p_JoinKernel->GetResultEventBuffer();
}

int GpuJoinProcessor::GetResultEventBufferSize()
{
	return p_JoinKernel->GetResultEventBufferSize();
}

}


