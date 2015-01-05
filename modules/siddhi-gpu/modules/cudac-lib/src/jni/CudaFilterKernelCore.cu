/*
 * CudaFilterKernelCore.cu
 *
 *  Created on: Nov 9, 2014
 *      Author: prabodha
 */

#ifndef CUDAFILTERKERNELCORE_CU_
#define CUDAFILTERKERNELCORE_CU_

#include "CudaFilterKernelCore.h"
#include <limits.h>
#include <float.h>

namespace SiddhiGpu
{

__device__ bool cuda_strcmp(const char *s1, const char *s2)
{
	for ( ; *s1==*s2; ++s1, ++s2) {
		if (*s1=='\0') return true;
	}
	return false;
}

__device__ bool cuda_prefix(char *s1, char *s2)
{
	for ( ; *s1==*s2; ++s1, ++s2) {
		if (*(s2+1)=='\0') return true;
	}
	return false;
}

__device__ bool cuda_contains(const char *s1, const char *s2)
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

__device__ int AddExpressionInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int MinExpressionInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int MulExpressionInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int DivExpressionInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int ModExpressionInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) %
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

// ========================= LONG ==============================================

__device__ int64_t AddExpressionLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int64_t MinExpressionLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int64_t MulExpressionLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int64_t DivExpressionLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int64_t ModExpressionLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) %
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}


// ========================= FLOAT ==============================================

__device__ float AddExpressionFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ float MinExpressionFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ float MulExpressionFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ float DivExpressionFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ float ModExpressionFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
//	return ((int64_t)ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) %
//			(int64_t)ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));

	return fmod(ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex),
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

// ========================= DOUBLE ===========================================

__device__ double AddExpressionDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ double MinExpressionDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ double MulExpressionDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ double DivExpressionDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ double ModExpressionDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
//	return ((int64_t)ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) %
//			(int64_t)ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));

	return fmod(ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex),
				ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

/// ============================ CONDITION EXECUTORS ==========================

__device__ bool InvalidOperator(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return false;
}

__device__ bool NoopOperator(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return true;
}

// Equal operators

__device__ bool EqualCompareBoolBool(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareIntInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareIntLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareIntFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareIntDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareLongInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareLongLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareLongFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareLongDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareStringString(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (cuda_strcmp(ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex), ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex)));
}

//__device__ bool EqualCompareExecutorExecutor(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
//{
//	switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex + 1].e_NodeType)
//	{
//		case EXECUTOR_NODE_CONST:
//		{
//			switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex + 1].m_ConstValue.e_Type)
//			{
//			case INT:
//			{
////				return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex++) ==
////						ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex++));
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
////			return ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex++);
//		default:
//			break;
//	}
////	return (Execute(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex) ==
////			Execute(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex));
//}

/// ============================================================================
// NotEqual operator

__device__ bool NotEqualCompareBoolBool(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareStringString(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (!cuda_strcmp(ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex),ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex)));
}

/// ============================================================================

// GreaterThan operator

__device__ bool GreaterThanCompareIntInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareIntLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareIntFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareIntDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}


/// ============================================================================
// LessThan operator

__device__ bool LessThanCompareIntInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareIntLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareIntFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareIntDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

/// ============================================================================
// GreaterAndEqual operator

__device__ bool GreaterAndEqualCompareIntInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareIntLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareIntFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareIntDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

/// ============================================================================
// LessAndEqual operator

__device__ bool LessAndEqualCompareIntInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareIntLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareIntFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareIntDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleInt(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleLong(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleFloat(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleDouble(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}


__device__ bool ContainsOperator(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (cuda_contains(ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex), ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex)));
}


/// ============================================================================

__device__ bool ExecuteBoolExpression(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
					_pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Boolean)
			{
				// get attribute value
				int16_t i;
				memcpy(&i, _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 2);
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

__device__ int ExecuteIntExpression(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
					_pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Int)
			{
				int32_t i;
				memcpy(&i, _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);
				_iCurrentNodeIndex++;
				return i;
			}
		}
		break;
		case EXPRESSION_ADD_INT:
		{
			return AddExpressionInt(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_INT:
		{
			return MinExpressionInt(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_INT:
		{
			return MulExpressionInt(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_INT:
		{
			return DivExpressionInt(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_INT:
		{
			return ModExpressionInt(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return INT_MIN;
}

__device__ int64_t ExecuteLongExpression(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
					_pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Long)
			{
				int64_t i;
				memcpy(&i, _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);
				_iCurrentNodeIndex++;
				return i;
			}
		}
		break;
		case EXPRESSION_ADD_LONG:
		{
			return AddExpressionLong(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_LONG:
		{
			return MinExpressionLong(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_LONG:
		{
			return MulExpressionLong(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_LONG:
		{
			return DivExpressionLong(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_LONG:
		{
			return ModExpressionLong(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return LLONG_MIN;
}

__device__ float ExecuteFloatExpression(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
					_pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Float)
			{
				float f;
				memcpy(&f, _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);
				_iCurrentNodeIndex++;
				return f;
			}
		}
		break;
		case EXPRESSION_ADD_FLOAT:
		{
			return AddExpressionFloat(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_FLOAT:
		{
			return MinExpressionFloat(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_FLOAT:
		{
			return MulExpressionFloat(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_FLOAT:
		{
			return DivExpressionFloat(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_FLOAT:
		{
			return ModExpressionFloat(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return FLT_MIN;
}

__device__ double ExecuteDoubleExpression(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
					_pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Double)
			{
				double f;
				memcpy(&f, _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);
				_iCurrentNodeIndex++;
				return f;
			}
		}
		break;
		case EXPRESSION_ADD_DOUBLE:
		{
			return AddExpressionDouble(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_DOUBLE:
		{
			return MinExpressionDouble(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_DOUBLE:
		{
			return MulExpressionDouble(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_DOUBLE:
		{
			return DivExpressionDouble(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_DOUBLE:
		{
			return ModExpressionDouble(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return DBL_MIN;
}

__device__ const char * ExecuteStringExpression(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
					_pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::StringIn)
			{
				int16_t i;
				memcpy(&i, _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 2);
				char * z = _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position + 2;
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

__device__ bool AndCondition(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) && Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool OrCondition(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) || Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotCondition(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (!Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool BooleanCondition(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

// =========================================

// set all the executor functions here
__device__ ExecutorFuncPointer mExecutors[EXECUTOR_CONDITION_COUNT] = {
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
__device__ bool Evaluate(Filter & _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (*mExecutors[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].e_ConditionType])(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
}

};

#endif


