/*
 * CpuFilterKernel.cpp
 *
 *  Created on: Dec 28, 2014
 *      Author: prabodha
 */

#include "CpuFilterKernel.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>

namespace SiddhiCpu
{

bool cuda_strcmp(const char *s1, const char *s2)
{
	for ( ; *s1==*s2; ++s1, ++s2) {
		if (*s1=='\0') return true;
	}
	return false;
}

bool cuda_prefix(char *s1, char *s2)
{
	for ( ; *s1==*s2; ++s1, ++s2) {
		if (*(s2+1)=='\0') return true;
	}
	return false;
}

bool cuda_contains(const char *s1, const char *s2)
{
	int size1 = 0;
	int size2 = 0;

	while (s1[size1]!='\0')
		size1++;

	while (s2[size2]!='\0')
		size2++;

	if (size1==size2)
		return cuda_strcmp(s1, s2);

	if (size1<size2)
		return false;

	for (int i=0; i<size1-size2+1; i++)
	{
		bool failed = false;
		for (int j=0; j<size2; j++)
		{
			if (s1[i+j-1]!=s2[j])
			{
				failed = true;
				break;
			}
		}
		if (! failed)
			return true;
	}
	return false;
}

// ========================= INT ==============================================

int AddExpressionInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int MinExpressionInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int MulExpressionInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int DivExpressionInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int ModExpressionInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) %
			ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

// ========================= LONG ==============================================

int64_t AddExpressionLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int64_t MinExpressionLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int64_t MulExpressionLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int64_t DivExpressionLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int64_t ModExpressionLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) %
			ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}


// ========================= FLOAT ==============================================

float AddExpressionFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

float MinExpressionFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

float MulExpressionFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

float DivExpressionFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

float ModExpressionFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
//	return ((int64_t)ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) %
//			(int64_t)ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));

	return fmod(ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex),
			ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

// ========================= DOUBLE ===========================================

double AddExpressionDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

double MinExpressionDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

double MulExpressionDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

double DivExpressionDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

double ModExpressionDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
//	return ((int64_t)ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) %
//			(int64_t)ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));

	return fmod(ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex),
				ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

/// ============================ CONDITION EXECUTORS ==========================

bool InvalidOperator(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return false;
}

bool NoopOperator(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return true;
}

// Equal operators

bool EqualCompareBoolBool(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareIntInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareIntLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareIntFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareIntDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareLongInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareLongLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareLongFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareLongDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareFloatInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareFloatLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareFloatFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareFloatDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareDoubleInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareDoubleLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareDoubleFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareDoubleDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareStringString(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (cuda_strcmp(ExecuteStringExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex), ExecuteStringExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex)));
}

//bool EqualCompareExecutorExecutor(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
//{
//	switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex + 1].e_NodeType)
//	{
//		case EXECUTOR_NODE_CONST:
//		{
//			switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex + 1].m_ConstValue.e_Type)
//			{
//			case INT:
//			{
////				return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex++) ==
////						ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex++));
//			}
//			break;
//			case LONG:
//			case FLOAT:
//			case DOUBLE:
//			case STRING:
//			}
//
//			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DOUBLE)
//			{
//				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.d_DoubleVal;
//			}
//		}
//		break;
//		case EXECUTOR_NODE_VARIABLE:
//		{
//			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DOUBLE &&
//					_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition < _mEvent.i_NumAttributes &&
//					_mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].e_Type == DOUBLE)
//			{
//				return _mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_VarValue.i_AttributePosition].m_Value.d_DoubleVal;
//			}
//		}
//		break;
//		case EXECUTOR_NODE_CONDITION:
////			return ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex++);
//		default:
//			break;
//	}
////	return (Execute(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex) ==
////			Execute(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex));
//}

/// ============================================================================
// NotEqual operator

bool NotEqualCompareBoolBool(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareIntInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareIntLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareIntFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareIntDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareLongInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareLongLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareLongFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareLongDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareFloatInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareFloatLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareFloatFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareFloatDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareDoubleInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareDoubleLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareDoubleFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareDoubleDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareStringString(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (!cuda_strcmp(ExecuteStringExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex),ExecuteStringExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex)));
}

/// ============================================================================

// GreaterThan operator

bool GreaterThanCompareIntInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareIntLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareIntFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareIntDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareLongInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareLongLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareLongFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareLongDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareFloatInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareFloatLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareFloatFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareFloatDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareDoubleInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareDoubleLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareDoubleFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareDoubleDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}


/// ============================================================================
// LessThan operator

bool LessThanCompareIntInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareIntLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareIntFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareIntDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareLongInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareLongLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareLongFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareLongDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareFloatInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareFloatLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareFloatFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareFloatDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareDoubleInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareDoubleLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareDoubleFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareDoubleDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

/// ============================================================================
// GreaterAndEqual operator

bool GreaterAndEqualCompareIntInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareIntLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareIntFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareIntDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareLongInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareLongLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareLongFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareLongDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareFloatInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareFloatLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareFloatFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareFloatDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareDoubleInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareDoubleLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareDoubleFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareDoubleDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

/// ============================================================================
// LessAndEqual operator

bool LessAndEqualCompareIntInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareIntLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareIntFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareIntDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareLongInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareLongLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareLongFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareLongDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareFloatInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareFloatLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareFloatFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareFloatDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareDoubleInt(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareDoubleLong(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareDoubleFloat(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareDoubleDouble(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}


bool ContainsOperator(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (cuda_contains(ExecuteStringExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex), ExecuteStringExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex)));
}


/// ============================================================================

bool ExecuteBoolExpression(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::Boolean)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.b_BoolVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			// if filter data type matches event attribute data type, return attribute value
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == SiddhiGpu::DataType::Boolean &&
					_mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Boolean)
			{
				// get attribute value
				int16_t i;
				memcpy(&i, _pEvent + _mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 2);
				_iCurrentNodeIndex++;
				return i;
			}
		}
		break;
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return false;
}

int ExecuteIntExpression(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::Int)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.i_IntVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == SiddhiGpu::DataType::Int &&
					_mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Int)
			{
				int32_t i;
				memcpy(&i, _pEvent + _mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);
				_iCurrentNodeIndex++;
				return i;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_ADD_INT:
		{
			return AddExpressionInt(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_SUB_INT:
		{
			return MinExpressionInt(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_MUL_INT:
		{
			return MulExpressionInt(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_DIV_INT:
		{
			return DivExpressionInt(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_MOD_INT:
		{
			return ModExpressionInt(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return INT_MIN;
}

int64_t ExecuteLongExpression(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::Long)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.l_LongVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == SiddhiGpu::DataType::Long &&
					_mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Long)
			{
				int64_t i;
				memcpy(&i, _pEvent + _mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);
				_iCurrentNodeIndex++;
				return i;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_ADD_LONG:
		{
			return AddExpressionLong(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_SUB_LONG:
		{
			return MinExpressionLong(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_MUL_LONG:
		{
			return MulExpressionLong(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_DIV_LONG:
		{
			return DivExpressionLong(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_MOD_LONG:
		{
			return ModExpressionLong(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return LLONG_MIN;
}

float ExecuteFloatExpression(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::Float)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.f_FloatVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == SiddhiGpu::DataType::Float &&
					_mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Float)
			{
				float f;
				memcpy(&f, _pEvent + _mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);
				_iCurrentNodeIndex++;
				return f;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_ADD_FLOAT:
		{
			return AddExpressionFloat(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_SUB_FLOAT:
		{
			return MinExpressionFloat(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_MUL_FLOAT:
		{
			return MulExpressionFloat(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_DIV_FLOAT:
		{
			return DivExpressionFloat(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_MOD_FLOAT:
		{
			return ModExpressionFloat(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return FLT_MIN;
}

double ExecuteDoubleExpression(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::Double)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.d_DoubleVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == SiddhiGpu::DataType::Double &&
					_mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Double)
			{
				double f;
				memcpy(&f, _pEvent + _mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);
				_iCurrentNodeIndex++;
				return f;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_ADD_DOUBLE:
		{
			return AddExpressionDouble(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_SUB_DOUBLE:
		{
			return MinExpressionDouble(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_MUL_DOUBLE:
		{
			return MulExpressionDouble(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_DIV_DOUBLE:
		{
			return DivExpressionDouble(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case SiddhiGpu::EXPRESSION_MOD_DOUBLE:
		{
			return ModExpressionDouble(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return DBL_MIN;
}

const char * ExecuteStringExpression(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::StringIn)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.z_StringVal;
			}
			else if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::StringExt)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.z_ExtString;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == SiddhiGpu::DataType::StringIn &&
					_mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::StringIn)
			{
				int16_t i;
				memcpy(&i, _pEvent + _mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 2);
				char * z = _pEvent + _mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position + 2;
				z[i] = 0;
				_iCurrentNodeIndex++;
				return z;
			}
		}
		break;
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return NULL;
}

// ==================================================================================================

bool AndCondition(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) && Evaluate(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool OrCondition(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) || Evaluate(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotCondition(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (!Evaluate(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool BooleanCondition(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

// =========================================

// set all the executor functions here
ExecutorFuncPointer mExecutors[SiddhiGpu::EXECUTOR_CONDITION_COUNT] = {
		NoopOperator,

		AndCondition,
		OrCondition,
		NotCondition,
		BooleanCondition,

		EqualCompareBoolBool,
		EqualCompareIntInt,
		EqualCompareIntLong,
		EqualCompareIntFloat,
		EqualCompareIntDouble,
		EqualCompareLongInt,
		EqualCompareLongLong,
		EqualCompareLongFloat,
		EqualCompareLongDouble,
		EqualCompareFloatInt,
		EqualCompareFloatLong,
		EqualCompareFloatFloat,
		EqualCompareFloatDouble,
		EqualCompareDoubleInt,
		EqualCompareDoubleLong,
		EqualCompareDoubleFloat,
		EqualCompareDoubleDouble,
		EqualCompareStringString,

		NotEqualCompareBoolBool,
		NotEqualCompareIntInt,
		NotEqualCompareIntLong,
		NotEqualCompareIntFloat,
		NotEqualCompareIntDouble,
		NotEqualCompareLongInt,
		NotEqualCompareLongLong,
		NotEqualCompareLongFloat,
		NotEqualCompareLongDouble,
		NotEqualCompareFloatInt,
		NotEqualCompareFloatLong,
		NotEqualCompareFloatFloat,
		NotEqualCompareFloatDouble,
		NotEqualCompareDoubleInt,
		NotEqualCompareDoubleLong,
		NotEqualCompareDoubleFloat,
		NotEqualCompareDoubleDouble,
		NotEqualCompareStringString,

		GreaterThanCompareIntInt,
		GreaterThanCompareIntLong,
		GreaterThanCompareIntFloat,
		GreaterThanCompareIntDouble,
		GreaterThanCompareLongInt,
		GreaterThanCompareLongLong,
		GreaterThanCompareLongFloat,
		GreaterThanCompareLongDouble,
		GreaterThanCompareFloatInt,
		GreaterThanCompareFloatLong,
		GreaterThanCompareFloatFloat,
		GreaterThanCompareFloatDouble,
		GreaterThanCompareDoubleInt,
		GreaterThanCompareDoubleLong,
		GreaterThanCompareDoubleFloat,
		GreaterThanCompareDoubleDouble,

		LessThanCompareIntInt,
		LessThanCompareIntLong,
		LessThanCompareIntFloat,
		LessThanCompareIntDouble,
		LessThanCompareLongInt,
		LessThanCompareLongLong,
		LessThanCompareLongFloat,
		LessThanCompareLongDouble,
		LessThanCompareFloatInt,
		LessThanCompareFloatLong,
		LessThanCompareFloatFloat,
		LessThanCompareFloatDouble,
		LessThanCompareDoubleInt,
		LessThanCompareDoubleLong,
		LessThanCompareDoubleFloat,
		LessThanCompareDoubleDouble,

		GreaterAndEqualCompareIntInt,
		GreaterAndEqualCompareIntLong,
		GreaterAndEqualCompareIntFloat,
		GreaterAndEqualCompareIntDouble,
		GreaterAndEqualCompareLongInt,
		GreaterAndEqualCompareLongLong,
		GreaterAndEqualCompareLongFloat,
		GreaterAndEqualCompareLongDouble,
		GreaterAndEqualCompareFloatInt,
		GreaterAndEqualCompareFloatLong,
		GreaterAndEqualCompareFloatFloat,
		GreaterAndEqualCompareFloatDouble,
		GreaterAndEqualCompareDoubleInt,
		GreaterAndEqualCompareDoubleLong,
		GreaterAndEqualCompareDoubleFloat,
		GreaterAndEqualCompareDoubleDouble,

		LessAndEqualCompareIntInt,
		LessAndEqualCompareIntLong,
		LessAndEqualCompareIntFloat,
		LessAndEqualCompareIntDouble,
		LessAndEqualCompareLongInt,
		LessAndEqualCompareLongLong,
		LessAndEqualCompareLongFloat,
		LessAndEqualCompareLongDouble,
		LessAndEqualCompareFloatInt,
		LessAndEqualCompareFloatLong,
		LessAndEqualCompareFloatFloat,
		LessAndEqualCompareFloatDouble,
		LessAndEqualCompareDoubleInt,
		LessAndEqualCompareDoubleLong,
		LessAndEqualCompareDoubleFloat,
		LessAndEqualCompareDoubleDouble,

		ContainsOperator,

		InvalidOperator,
};

// =========================================

// evaluate event with an executor tree
bool Evaluate(SiddhiGpu::Filter& _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (*mExecutors[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].e_ConditionType])(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
}

};



