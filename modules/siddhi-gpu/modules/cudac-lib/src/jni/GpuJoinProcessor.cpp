/*
 * GpuJoinProcessor.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: prabodha
 */

#include "GpuMetaEvent.h"
#include "GpuJoinKernel.h"
#include "GpuProcessorContext.h"
#include "GpuFilterProcessor.h"
#include "GpuJoinProcessor.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

namespace SiddhiGpu
{

GpuJoinProcessor::GpuJoinProcessor(int _iLeftWindowSize, int _iRightWindowSize) :
	GpuProcessor(GpuProcessor::JOIN),
	i_NodeCount(0),
	ap_ExecutorNodes(NULL),
	i_LeftStraemWindowSize(_iLeftWindowSize),
	i_RightStraemWindowSize(_iRightWindowSize),
	p_LeftStreamMetaEvent(NULL),
	p_RightStreamMetaEvent(NULL),
	b_LeftTrigger(true),
	b_RightTrigger(true),
	i_WithInTimeMilliSeconds(0),
	p_LeftContext(NULL),
	p_RightContext(NULL),
	p_JoinKernel(NULL),
	p_LeftPrevProcessor(NULL),
	p_RightPrevProcessor(NULL),
	fp_LeftLog(NULL),
	fp_RightLog(NULL)
{
}

GpuJoinProcessor::~GpuJoinProcessor()
{
	if(p_JoinKernel)
	{
		delete p_JoinKernel;
		p_JoinKernel = NULL;
	}

	if(ap_ExecutorNodes)
	{
		delete [] ap_ExecutorNodes;
		ap_ExecutorNodes = NULL;
	}

	p_LeftContext = NULL;
	p_RightContext = NULL;

	p_LeftPrevProcessor = NULL;
	p_RightPrevProcessor = NULL;

	p_LeftStreamMetaEvent = NULL;
	p_RightStreamMetaEvent = NULL;

}

GpuProcessor * GpuJoinProcessor::Clone()
{
	GpuJoinProcessor * pCloned = new GpuJoinProcessor(i_LeftStraemWindowSize, i_RightStraemWindowSize);

	return pCloned;
}

void GpuJoinProcessor::Configure(int _iStreamIndex, GpuProcessor * _pPrevProcessor, GpuProcessorContext * _pContext, FILE * _fpLog)
{
	fprintf(_fpLog, "[GpuJoinProcessor] Configure : StreamIndex=%d PrevProcessor=%p Context=%p \n", _iStreamIndex, _pPrevProcessor, _pContext);
	fflush(_fpLog);

	if(_iStreamIndex == 0)
	{
		p_LeftPrevProcessor = _pPrevProcessor;
		p_LeftContext = _pContext;
		fp_LeftLog = _fpLog;
	}
	else if(_iStreamIndex == 1)
	{
		p_RightPrevProcessor = _pPrevProcessor;
		p_RightContext = _pContext;
		fp_RightLog = _fpLog;
	}
}

void GpuJoinProcessor::Init(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize)
{
	if(_iStreamIndex == 0)
	{
		fprintf(fp_LeftLog, "[GpuJoinProcessor] Init : StreamIndex=%d\n", _iStreamIndex);
		fflush(fp_LeftLog);
	}
	else if(_iStreamIndex == 1)
	{
		fprintf(fp_RightLog, "[GpuJoinProcessor] Init : StreamIndex=%d\n", _iStreamIndex);
		fflush(fp_RightLog);
	}

	if(p_JoinKernel == NULL)
	{
		p_JoinKernel = new GpuJoinKernel(this, p_LeftContext, p_RightContext, i_ThreadBlockSize,
				i_LeftStraemWindowSize, i_RightStraemWindowSize, fp_LeftLog, fp_RightLog);
		p_JoinKernel->SetLeftInputEventBufferIndex(0);
		p_JoinKernel->SetRightInputEventBufferIndex(0);
		if(p_Next == NULL)
		{
			p_JoinKernel->SetOutputStream(p_OutputStreamMeta, p_OutputAttributeMapping);
		}
	}

	if(p_LeftPrevProcessor)
	{
		switch(p_LeftPrevProcessor->GetType())
		{
		case GpuProcessor::FILTER:
		{
			p_JoinKernel->SetLeftInputEventBufferIndex(p_LeftPrevProcessor->GetResultEventBufferIndex());
			p_JoinKernel->SetLeftFirstKernel(false);
		}
		break;
		default:
			break;
		}
	}

	if(p_RightPrevProcessor)
	{
		switch(p_RightPrevProcessor->GetType())
		{
		case GpuProcessor::FILTER:
		{
			p_JoinKernel->SetRightInputEventBufferIndex(p_RightPrevProcessor->GetResultEventBufferIndex());
			p_JoinKernel->SetRightFirstKernel(false);
		}
		break;
		default:
			break;
		}
	}

	p_JoinKernel->Initialize(_iStreamIndex, _pMetaEvent, _iInputEventBufferSize);

}

int GpuJoinProcessor::Process(int _iStreamIndex, int _iNumEvents)
{
#ifdef GPU_DEBUG
	if(_iStreamIndex == 0)
	{
		fprintf(fp_LeftLog, "[GpuJoinProcessor] Process : StreamIndex=%d NumEvents=%d \n", _iStreamIndex, _iNumEvents);
		fflush(fp_LeftLog);
	}
	else if(_iStreamIndex == 1)
	{
		fprintf(fp_RightLog, "[GpuJoinProcessor] Process : StreamIndex=%d NumEvents=%d \n", _iStreamIndex, _iNumEvents);
		fflush(fp_RightLog);
	}
#endif

	p_JoinKernel->Process(_iStreamIndex, _iNumEvents);

	if(p_Next)
	{
		_iNumEvents = p_Next->Process(_iStreamIndex, _iNumEvents);
	}


	return _iNumEvents;
}

void GpuJoinProcessor::Print(FILE * _fp)
{
	if(p_LeftStreamMetaEvent)
	{
		fprintf(_fp, "[GpuJoinProcessor] MetaStreams : Left={StreamIndex=%d|AttrCount=%d|EventSize=%d} \n",
				p_LeftStreamMetaEvent->i_StreamIndex, p_LeftStreamMetaEvent->i_AttributeCount, p_LeftStreamMetaEvent->i_SizeOfEventInBytes);
	}
	if(p_RightStreamMetaEvent)
	{
		fprintf(_fp, "[GpuJoinProcessor] MetaStreams : Right={StreamIndex=%d|AttrCount=%d|EventSize=%d} \n",
				p_RightStreamMetaEvent->i_StreamIndex, p_RightStreamMetaEvent->i_AttributeCount, p_RightStreamMetaEvent->i_SizeOfEventInBytes);
	}
	fprintf(_fp, "[GpuJoinProcessor] WindowSize : Left=%d Right=%d \n", i_LeftStraemWindowSize, i_RightStraemWindowSize);
	fprintf(_fp, "[GpuJoinProcessor] Trigger : Left=%d Right=%d \n", b_LeftTrigger, b_RightTrigger);
	fprintf(_fp, "[GpuJoinProcessor] WithIn : %" PRIi64 " milliseconds \n", i_WithInTimeMilliSeconds);
	fprintf(_fp, "[GpuJoinProcessor] OnCompare : [%p] NodeCount=%d {", this, i_NodeCount);
	for(int i=0; i<i_NodeCount; ++i)
	{
		ap_ExecutorNodes[i].Print(_fp);
		fprintf(_fp, "|");
	}
	fprintf(_fp, "}\n");
	fflush(_fp);
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

void GpuJoinProcessor::SetExecutorNodes(int _iNodeCount)
{
	i_NodeCount = _iNodeCount;
	ap_ExecutorNodes = new ExecutorNode[i_NodeCount];
}

void GpuJoinProcessor::AddExecutorNode(int _iPos, ExecutorNode & _pNode)
{
	if(_iPos < i_NodeCount)
	{
		ap_ExecutorNodes[_iPos].e_NodeType = _pNode.e_NodeType;
		ap_ExecutorNodes[_iPos].e_ConditionType = _pNode.e_ConditionType;
		ap_ExecutorNodes[_iPos].e_ExpressionType = _pNode.e_ExpressionType;
		ap_ExecutorNodes[_iPos].m_ConstValue.e_Type = _pNode.m_ConstValue.e_Type;
		ap_ExecutorNodes[_iPos].m_ConstValue.m_Value = _pNode.m_ConstValue.m_Value;
		ap_ExecutorNodes[_iPos].m_VarValue.i_StreamIndex = _pNode.m_VarValue.i_StreamIndex;
		ap_ExecutorNodes[_iPos].m_VarValue.e_Type = _pNode.m_VarValue.e_Type;
		ap_ExecutorNodes[_iPos].m_VarValue.i_AttributePosition = _pNode.m_VarValue.i_AttributePosition;
	}
	else
	{
		printf("[ERROR] [GpuJoinProcessor::AddExecutorNode] array out of bound : %d >= %d\n", _iPos, i_NodeCount);
	}
}

char * GpuJoinProcessor::GetLeftResultEventBuffer()
{
	return p_JoinKernel->GetLeftResultEventBuffer();
}

int GpuJoinProcessor::GetLeftResultEventBufferSize()
{
	return p_JoinKernel->GetLeftResultEventBufferSize();
}

char * GpuJoinProcessor::GetRightResultEventBuffer()
{
	return p_JoinKernel->GetRightResultEventBuffer();
}

int GpuJoinProcessor::GetRightResultEventBufferSize()
{
	return p_JoinKernel->GetRightResultEventBufferSize();
}

}


