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

namespace SiddhiGpu
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

int AddExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int MinExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int MulExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int DivExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int ModExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) %
			ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

// ========================= LONG ==============================================

int64_t AddExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int64_t MinExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int64_t MulExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int64_t DivExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

int64_t ModExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) %
			ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}


// ========================= FLOAT ==============================================

float AddExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

float MinExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

float MulExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

float DivExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

float ModExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
//	return ((int64_t)ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) %
//			(int64_t)ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));

	return fmod(ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex),
			ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

// ========================= DOUBLE ===========================================

double AddExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

double MinExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

double MulExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

double DivExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

double ModExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
//	return ((int64_t)ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) %
//			(int64_t)ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));

	return fmod(ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex),
				ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

/// ============================ CONDITION EXECUTORS ==========================

bool InvalidOperator(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return false;
}

bool NoopOperator(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return true;
}

// Equal operators

bool EqualCompareBoolBool(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool EqualCompareStringString(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (cuda_strcmp(ExecuteStringExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex), ExecuteStringExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex)));
}

//bool EqualCompareExecutorExecutor(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
////	return (Execute(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex) ==
////			Execute(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex));
//}

/// ============================================================================
// NotEqual operator

bool NotEqualCompareBoolBool(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotEqualCompareStringString(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (!cuda_strcmp(ExecuteStringExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex),ExecuteStringExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex)));
}

/// ============================================================================

// GreaterThan operator

bool GreaterThanCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterThanCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}


/// ============================================================================
// LessThan operator

bool LessThanCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessThanCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

/// ============================================================================
// GreaterAndEqual operator

bool GreaterAndEqualCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool GreaterAndEqualCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

/// ============================================================================
// LessAndEqual operator

bool LessAndEqualCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool LessAndEqualCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}


bool ContainsOperator(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (cuda_contains(ExecuteStringExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex), ExecuteStringExpression(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex)));
}


/// ============================================================================

bool ExecuteBoolExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::Boolean)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.b_BoolVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			// if filter data type matches event attribute data type, return attribute value
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Boolean &&
					_mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Boolean)
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

int ExecuteIntExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::Int)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.i_IntVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Int &&
					_mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Int)
			{
				int32_t i;
				memcpy(&i, _pEvent + _mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);
				_iCurrentNodeIndex++;
				return i;
			}
		}
		break;
		case EXPRESSION_ADD_INT:
		{
			return AddExpressionInt(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_INT:
		{
			return MinExpressionInt(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_INT:
		{
			return MulExpressionInt(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_INT:
		{
			return DivExpressionInt(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_INT:
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

int64_t ExecuteLongExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::Long)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.l_LongVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Long &&
					_mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Long)
			{
				int64_t i;
				memcpy(&i, _pEvent + _mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);
				_iCurrentNodeIndex++;
				return i;
			}
		}
		break;
		case EXPRESSION_ADD_LONG:
		{
			return AddExpressionLong(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_LONG:
		{
			return MinExpressionLong(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_LONG:
		{
			return MulExpressionLong(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_LONG:
		{
			return DivExpressionLong(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_LONG:
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

float ExecuteFloatExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::Float)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.f_FloatVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Float &&
					_mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Float)
			{
				float f;
				memcpy(&f, _pEvent + _mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);
				_iCurrentNodeIndex++;
				return f;
			}
		}
		break;
		case EXPRESSION_ADD_FLOAT:
		{
			return AddExpressionFloat(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_FLOAT:
		{
			return MinExpressionFloat(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_FLOAT:
		{
			return MulExpressionFloat(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_FLOAT:
		{
			return DivExpressionFloat(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_FLOAT:
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

double ExecuteDoubleExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::Double)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.d_DoubleVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Double &&
					_mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Double)
			{
				double f;
				memcpy(&f, _pEvent + _mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);
				_iCurrentNodeIndex++;
				return f;
			}
		}
		break;
		case EXPRESSION_ADD_DOUBLE:
		{
			return AddExpressionDouble(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_DOUBLE:
		{
			return MinExpressionDouble(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_DOUBLE:
		{
			return MulExpressionDouble(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_DOUBLE:
		{
			return DivExpressionDouble(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_DOUBLE:
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

const char * ExecuteStringExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::StringIn)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.z_StringVal;
			}
			else if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::StringExt)
			{
				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.z_ExtString;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::StringIn &&
					_mEventMeta.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::StringIn)
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

bool AndCondition(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) && Evaluate(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool OrCondition(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex) || Evaluate(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool NotCondition(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (!Evaluate(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

bool BooleanCondition(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex));
}

// =========================================

// set all the executor functions here
ExecutorFuncPointer mExecutors[EXECUTOR_CONDITION_COUNT] = {
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
bool Evaluate(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (*mExecutors[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].e_ConditionType])(_mFilter, _mEventMeta, _pEvent, _iCurrentNodeIndex);
}

};



