/*
 * GpuFilterProcessor.cpp
 *
 *  Created on: Jan 20, 2015
 *      Author: prabodha
 */

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include "GpuFilterKernel.h"
#include "GpuProcessorContext.h"
#include "GpuFilterProcessor.h"

namespace SiddhiGpu
{

//VariableValue::VariableValue()
//{
//	i_StreamIndex = 0;
//	e_Type = DataType::None;
//	i_AttributePosition = -1;
//}
//
//VariableValue::VariableValue(int _iStreamIndex, DataType::Value _eType, int _iPos)
//{
//	i_StreamIndex = _iStreamIndex;
//	e_Type = _eType;
//	i_AttributePosition = _iPos;
//}

VariableValue & VariableValue::Init()
{
	i_StreamIndex = 0;
	e_Type = DataType::None;
	i_AttributePosition = -1;
	return *this;
}

VariableValue & VariableValue::SetStreamIndex(int _iStreamIndex)
{
	i_StreamIndex = _iStreamIndex;
	return *this;
}

VariableValue & VariableValue::SetDataType(DataType::Value _eType)
{
	e_Type = _eType;
	return *this;
}

VariableValue & VariableValue::SetPosition(int _iPos)
{
	i_AttributePosition = _iPos;
	return *this;
}

void VariableValue::Print(FILE * _fp)
{
	fprintf(_fp, "(%s-POS-%d:%d)", DataType::GetTypeName(e_Type), i_StreamIndex, i_AttributePosition);
}

// ============================================================================================================

//ConstValue::ConstValue()
//{
//	e_Type = DataType::None;
//}

ConstValue & ConstValue::Init()
{
	e_Type = DataType::None;
	return *this;
}

ConstValue & ConstValue::SetBool(bool _bVal)
{
	e_Type = DataType::Boolean;
	m_Value.b_BoolVal = _bVal;
	return *this;
}

ConstValue & ConstValue::SetInt(int _iVal)
{
	e_Type = DataType::Int;
	m_Value.i_IntVal = _iVal;
	return *this;
}

ConstValue & ConstValue::SetLong(int64_t _lVal)
{
	e_Type = DataType::Long;
	m_Value.l_LongVal = _lVal;
	return *this;
}

ConstValue & ConstValue::SetFloat(float _fval)
{
	e_Type = DataType::Float;
	m_Value.f_FloatVal = _fval;
	return *this;
}

ConstValue & ConstValue::SetDouble(double _dVal)
{
	e_Type = DataType::Double;
	m_Value.d_DoubleVal = _dVal;
	return *this;
}

ConstValue & ConstValue::SetString(const char * _zVal, int _iLen)
{
	if(_iLen > (int)sizeof(uint64_t))
	{
		e_Type = DataType::StringExt;
		m_Value.z_ExtString = new char[_iLen + 1];
		strncpy(m_Value.z_ExtString, _zVal, _iLen);
	}
	else
	{
		e_Type = DataType::StringIn;
		strncpy(m_Value.z_StringVal, _zVal, _iLen);
	}
	return *this;
}

void ConstValue::Print(FILE * _fp)
{
	fprintf(_fp, "(%s-", DataType::GetTypeName(e_Type));
	switch(e_Type)
	{
	case DataType::Int: //     = 0,
		fprintf(_fp, "%d)", m_Value.i_IntVal);
		break;
	case DataType::Long: //    = 1,
		fprintf(_fp, "%" PRIi64 ")", m_Value.l_LongVal);
		break;
	case DataType::Boolean: //    = 1,
		fprintf(_fp, "%d)", m_Value.b_BoolVal);
		break;
	case DataType::Float: //   = 2,
		fprintf(_fp, "%f)", m_Value.f_FloatVal);
		break;
	case DataType::Double://  = 3,
		fprintf(_fp, "%f)", m_Value.d_DoubleVal);
		break;
	case DataType::StringIn: //  = 4,
		fprintf(_fp, "%s)", m_Value.z_StringVal);
		break;
	case DataType::StringExt: //  = 4,
		fprintf(_fp, "%s)", m_Value.z_ExtString);
		break;
	case DataType::None: //    = 5
		fprintf(_fp, "NONE)");
		break;
	default:
		break;
	}
}

// ==========================================================================================================

//ExecutorNode::ExecutorNode()
//{
//	e_NodeType = EXECUTOR_NODE_TYPE_COUNT;
//	e_ConditionType = EXECUTOR_INVALID;
//	e_ExpressionType = EXPRESSION_INVALID;
//}

ExecutorNode & ExecutorNode::Init()
{
	e_NodeType = EXECUTOR_NODE_TYPE_COUNT;
	e_ConditionType = EXECUTOR_INVALID;
	e_ExpressionType = EXPRESSION_INVALID;
	return *this;
}

ExecutorNode & ExecutorNode::SetNodeType(ExecutorNodeType _eNodeType)
{
	e_NodeType = _eNodeType;
	return *this;
}

ExecutorNode & ExecutorNode::SetConditionType(ConditionType _eCondType)
{
	e_ConditionType = _eCondType;
	return *this;
}

ExecutorNode & ExecutorNode::SetExpressionType(ExpressionType _eExprType)
{
	e_ExpressionType = _eExprType;
	return *this;
}

ExecutorNode & ExecutorNode::SetConstValue(ConstValue _mConstVal)
{
	m_ConstValue.e_Type = _mConstVal.e_Type;
	m_ConstValue.m_Value = _mConstVal.m_Value;
	return *this;
}

ExecutorNode & ExecutorNode::SetVariableValue(VariableValue _mVarValue)
{
	m_VarValue.e_Type = _mVarValue.e_Type;
	m_VarValue.i_AttributePosition = _mVarValue.i_AttributePosition;
	m_VarValue.i_StreamIndex = _mVarValue.i_StreamIndex;
	return *this;
}

void ExecutorNode::Print(FILE * _fp)
{
	fprintf(_fp, "%s=", GetNodeTypeName(e_NodeType));
	switch(e_NodeType)
	{
	case EXECUTOR_NODE_CONDITION:
	{
		fprintf(_fp, "%s ", GetConditionName(e_ConditionType));
	}
	break;
	case EXECUTOR_NODE_EXPRESSION:
	{
		fprintf(_fp, "%s ", GetExpressionTypeName(e_ExpressionType));

		switch(e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			m_ConstValue.Print(_fp);
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			m_VarValue.Print(_fp);
		}
		break;
		default:
			break;
		}
	}
	break;
	default:
		break;
	}
}


// ==========================================================================================================

GpuFilterProcessor::GpuFilterProcessor(int _iNodeCount) :
	GpuProcessor(GpuProcessor::FILTER),
	i_NodeCount(_iNodeCount),
	p_Context(NULL),
	p_FilterKernel(NULL)
{
	ap_ExecutorNodes = new ExecutorNode[i_NodeCount];
}

GpuFilterProcessor::~GpuFilterProcessor()
{
	Destroy();
}

void GpuFilterProcessor::Destroy()
{
	if(ap_ExecutorNodes)
	{
		delete [] ap_ExecutorNodes;
		ap_ExecutorNodes = NULL;
	}

	if(p_FilterKernel)
	{
		delete p_FilterKernel;
		p_FilterKernel = NULL;
	}
}

void GpuFilterProcessor::AddExecutorNode(int _iPos, ExecutorNode & _pNode)
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
		printf("[ERROR] [GpuFilterProcessor::AddExecutorNode] array out of bound : %d >= %d\n", _iPos, i_NodeCount);
	}
}

GpuProcessor * GpuFilterProcessor::Clone()
{
	GpuFilterProcessor * f = new GpuFilterProcessor(i_NodeCount);

	memcpy(f->ap_ExecutorNodes, ap_ExecutorNodes, sizeof(ExecutorNode) * i_NodeCount);

	return f;
}

void GpuFilterProcessor::Configure(int _iStreamIndex, GpuProcessor * _pPrevProcessor, GpuProcessorContext * _pContext, FILE * _fpLog)
{
	fp_Log = _fpLog;
	p_Context = _pContext;

	fprintf(fp_Log, "[GpuFilterProcessor] Configure : StreamIndex=%d PrevProcessor=%p Context=%p \n", _iStreamIndex, _pPrevProcessor, p_Context);
	fflush(fp_Log);
}

void GpuFilterProcessor::Init(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize)
{
	fprintf(fp_Log, "[GpuFilterProcessor] Init : StreamIndex=%d DeviceId=%d InputEventBufferSize=%d\n",
			_iStreamIndex, p_Context->GetDeviceId(), _iInputEventBufferSize);
	fflush(fp_Log);

	if(p_Next)
	{
		p_FilterKernel = new GpuFilterKernelFirst(this, p_Context, i_ThreadBlockSize, fp_Log);
		p_FilterKernel->SetInputEventBufferIndex(0);

		fprintf(fp_Log, "[GpuFilterProcessor] Init : created GpuFilterKernelFirst \n");
		fflush(fp_Log);
	}
	else
	{
		p_FilterKernel = new GpuFilterKernelStandalone(this, p_Context, i_ThreadBlockSize, fp_Log);
		p_FilterKernel->SetInputEventBufferIndex(0);

		fprintf(fp_Log, "[GpuFilterProcessor] Init : created GpuFilterKernelStandalone \n");
		fflush(fp_Log);
	}

	if(p_Next == NULL)
	{
		p_FilterKernel->SetOutputStream(p_OutputStreamMeta, p_OutputAttributeMapping);
	}

	p_FilterKernel->Initialize(_iStreamIndex, _pMetaEvent, _iInputEventBufferSize);
}

int GpuFilterProcessor::Process(int _iStreamIndex, int _iNumEvents)
{
#if GPU_DEBUG >= GPU_DEBUG_LEVEL_TRACE
	fprintf(fp_Log, "[GpuFilterProcessor] Process : StreamIndex=%d NumEvents=%d \n", _iStreamIndex, _iNumEvents);
	fflush(fp_Log);
#endif
	// invoke kernels
	// get result meta data (resulting events count)
	p_FilterKernel->Process(_iStreamIndex, _iNumEvents);

	if(p_Next)
	{
		_iNumEvents = p_Next->Process(_iStreamIndex, _iNumEvents);
	}

	return _iNumEvents;
}

void GpuFilterProcessor::Print(FILE * _fp)
{
	fprintf(_fp, "[GpuFilterProcessor] AddFilter : [%p] NodeCount=%d {", this, i_NodeCount);
	for(int i=0; i<i_NodeCount; ++i)
	{
		ap_ExecutorNodes[i].Print(_fp);
		fprintf(_fp, "|");
	}
	fprintf(_fp, "}\n");
	fflush(_fp);
}

int GpuFilterProcessor::GetResultEventBufferIndex()
{
	return p_FilterKernel->GetResultEventBufferIndex();
}

char * GpuFilterProcessor::GetResultEventBuffer()
{
	fprintf(fp_Log, "[GpuFilterProcessor] GetResultEventBuffer : Kernel=%p Buffer=%p Size=%d \n",
			p_FilterKernel, p_FilterKernel->GetResultEventBuffer(), p_FilterKernel->GetResultEventBufferSize());
	fflush(fp_Log);
	return p_FilterKernel->GetResultEventBuffer();
}

int GpuFilterProcessor::GetResultEventBufferSize()
{
	return p_FilterKernel->GetResultEventBufferSize();
}

};


