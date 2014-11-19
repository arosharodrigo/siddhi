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

__device__ int AddExpressionInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mEvent, _mFilter, _iCurrentNodeIndex) +
			ExecuteIntExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ int MinExpressionInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mEvent, _mFilter, _iCurrentNodeIndex) -
			ExecuteIntExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ int MulExpressionInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mEvent, _mFilter, _iCurrentNodeIndex) *
			ExecuteIntExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ int DivExpressionInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mEvent, _mFilter, _iCurrentNodeIndex) /
			ExecuteIntExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ int ModExpressionInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mEvent, _mFilter, _iCurrentNodeIndex) %
			ExecuteIntExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

// ========================= LONG ==============================================

__device__ long AddExpressionLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mEvent, _mFilter, _iCurrentNodeIndex) +
			ExecuteLongExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ long MinExpressionLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mEvent, _mFilter, _iCurrentNodeIndex) -
			ExecuteLongExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ long MulExpressionLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mEvent, _mFilter, _iCurrentNodeIndex) *
			ExecuteLongExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ long DivExpressionLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mEvent, _mFilter, _iCurrentNodeIndex) /
			ExecuteLongExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ long ModExpressionLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mEvent, _mFilter, _iCurrentNodeIndex) %
			ExecuteLongExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}


// ========================= FLOAT ==============================================

__device__ float AddExpressionFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mEvent, _mFilter, _iCurrentNodeIndex) +
			ExecuteFloatExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ float MinExpressionFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mEvent, _mFilter, _iCurrentNodeIndex) -
			ExecuteFloatExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ float MulExpressionFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mEvent, _mFilter, _iCurrentNodeIndex) *
			ExecuteFloatExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ float DivExpressionFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mEvent, _mFilter, _iCurrentNodeIndex) /
			ExecuteFloatExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ float ModExpressionFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
//	return ((int64_t)ExecuteFloatExpression(_mEvent, _mFilter, _iCurrentNodeIndex) %
//			(int64_t)ExecuteFloatExpression(_mEvent, _mFilter, _iCurrentNodeIndex));

	return fmod(ExecuteFloatExpression(_mEvent, _mFilter, _iCurrentNodeIndex),
			ExecuteFloatExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

// ========================= DOUBLE ===========================================

__device__ double AddExpressionDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mEvent, _mFilter, _iCurrentNodeIndex) +
			ExecuteDoubleExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ double MinExpressionDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mEvent, _mFilter, _iCurrentNodeIndex) -
			ExecuteDoubleExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ double MulExpressionDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mEvent, _mFilter, _iCurrentNodeIndex) *
			ExecuteDoubleExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ double DivExpressionDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mEvent, _mFilter, _iCurrentNodeIndex) /
			ExecuteDoubleExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ double ModExpressionDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
//	return ((int64_t)ExecuteDoubleExpression(_mEvent, _mFilter, _iCurrentNodeIndex) %
//			(int64_t)ExecuteDoubleExpression(_mEvent, _mFilter, _iCurrentNodeIndex));

	return fmod(ExecuteDoubleExpression(_mEvent, _mFilter, _iCurrentNodeIndex),
				ExecuteDoubleExpression(_mEvent, _mFilter, _iCurrentNodeIndex));
}

/// ============================ CONDITION EXECUTORS ==========================

__device__ bool InvalidOperator(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return false;
}

__device__ bool NoopOperator(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return true;
}

// Equal operators

__device__ bool EqualCompareBoolBool(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareIntInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareIntLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareIntFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareIntDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareLongInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareLongLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareLongFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareLongDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) == ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool EqualCompareStringString(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (cuda_strcmp(ExecuteStringExpression(_mEvent, _mFilter, _iCurrentNodeIndex), ExecuteStringExpression(_mEvent, _mFilter, _iCurrentNodeIndex)));
}

//__device__ bool EqualCompareExecutorExecutor(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
//{
//	switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex + 1].e_NodeType)
//	{
//		case EXECUTOR_NODE_CONST:
//		{
//			switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex + 1].m_ConstValue.e_Type)
//			{
//			case INT:
//			{
////				return (ExecuteIntExpression(_mEvent, _mFilter, _iCurrentNodeIndex++) ==
////						ExecuteIntExpression(_mEvent, _mFilter, _iCurrentNodeIndex++));
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
////			return ExecuteDoubleExpression(_mEvent, _mFilter, _iCurrentNodeIndex++);
//		default:
//			break;
//	}
////	return (Execute(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex) ==
////			Execute(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex));
//}

/// ============================================================================
// NotEqual operator

__device__ bool NotEqualCompareBoolBool(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) != ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool NotEqualCompareStringString(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (!cuda_strcmp(ExecuteStringExpression(_mEvent, _mFilter,_iCurrentNodeIndex),ExecuteStringExpression(_mEvent, _mFilter,_iCurrentNodeIndex)));
}

/// ============================================================================

// GreaterThan operator

__device__ bool GreaterThanCompareIntInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareIntLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareIntFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareIntDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) > ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}


/// ============================================================================
// LessThan operator

__device__ bool LessThanCompareIntInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareIntLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareIntFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareIntDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) < ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

/// ============================================================================
// GreaterAndEqual operator

__device__ bool GreaterAndEqualCompareIntInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareIntLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareIntFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareIntDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) >= ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

/// ============================================================================
// LessAndEqual operator

__device__ bool LessAndEqualCompareIntInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareIntLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareIntFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareIntDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteIntExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteLongExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteFloatExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex) <= ExecuteDoubleExpression(_mEvent, _mFilter,_iCurrentNodeIndex));
}


__device__ bool ContainsOperator(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (cuda_contains(ExecuteStringExpression(_mEvent, _mFilter, _iCurrentNodeIndex), ExecuteStringExpression(_mEvent, _mFilter, _iCurrentNodeIndex)));
}


/// ============================================================================

__device__ bool ExecuteBoolExpression(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
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
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Boolean &&
					_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition < _mEvent.ui_NumAttributes &&
					_mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].e_Type == DataType::Boolean)
			{
				return _mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_VarValue.i_AttributePosition].m_Value.b_BoolVal;
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

__device__ int ExecuteIntExpression(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
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
					_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition < _mEvent.ui_NumAttributes &&
					_mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].e_Type == DataType::Int)
			{
				return _mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_VarValue.i_AttributePosition].m_Value.i_IntVal;
			}
		}
		break;
		case EXPRESSION_ADD_INT:
		{
			return AddExpressionInt(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_INT:
		{
			return MinExpressionInt(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_INT:
		{
			return MulExpressionInt(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_INT:
		{
			return DivExpressionInt(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_INT:
		{
			return ModExpressionInt(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return INT_MIN;
}

__device__ long ExecuteLongExpression(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
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
					_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition < _mEvent.ui_NumAttributes &&
					_mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].e_Type == DataType::Long)
			{
				return _mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_VarValue.i_AttributePosition].m_Value.l_LongVal;
			}
		}
		break;
		case EXPRESSION_ADD_LONG:
		{
			return AddExpressionLong(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_LONG:
		{
			return MinExpressionLong(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_LONG:
		{
			return MulExpressionLong(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_LONG:
		{
			return DivExpressionLong(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_LONG:
		{
			return ModExpressionLong(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return LLONG_MIN;
}

__device__ float ExecuteFloatExpression(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
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
					_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition < _mEvent.ui_NumAttributes &&
					_mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].e_Type == DataType::Float)
			{
				return _mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_VarValue.i_AttributePosition].m_Value.f_FloatVal;
			}
		}
		break;
		case EXPRESSION_ADD_FLOAT:
		{
			return AddExpressionFloat(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_FLOAT:
		{
			return MinExpressionFloat(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_FLOAT:
		{
			return MulExpressionFloat(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_FLOAT:
		{
			return DivExpressionFloat(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_FLOAT:
		{
			return ModExpressionFloat(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return FLT_MIN;
}

__device__ double ExecuteDoubleExpression(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
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
					_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition < _mEvent.ui_NumAttributes &&
					_mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].e_Type == DataType::Double)
			{
				return _mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_VarValue.i_AttributePosition].m_Value.d_DoubleVal;
			}
		}
		break;
		case EXPRESSION_ADD_DOUBLE:
		{
			return AddExpressionDouble(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_DOUBLE:
		{
			return MinExpressionDouble(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_DOUBLE:
		{
			return MulExpressionDouble(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_DOUBLE:
		{
			return DivExpressionDouble(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_DOUBLE:
		{
			return ModExpressionDouble(_mEvent, _mFilter, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return DBL_MIN;
}

__device__ const char * ExecuteStringExpression(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
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
					_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition < _mEvent.ui_NumAttributes &&
					_mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].e_Type == DataType::StringIn)
			{
				return _mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_VarValue.i_AttributePosition].m_Value.z_StringVal;
			}
			else if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::StringExt &&
					_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition < _mEvent.ui_NumAttributes &&
					_mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].e_Type == DataType::StringExt)
			{
				return _mEvent.a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_VarValue.i_AttributePosition].m_Value.z_ExtString;
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

__device__ bool AndCondition(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mEvent, _mFilter, _iCurrentNodeIndex) && Evaluate(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ bool OrCondition(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mEvent, _mFilter, _iCurrentNodeIndex) || Evaluate(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ bool NotCondition(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (!Evaluate(_mEvent, _mFilter, _iCurrentNodeIndex));
}

__device__ bool BooleanCondition(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mEvent, _mFilter, _iCurrentNodeIndex));
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
__device__ bool Evaluate(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex)
{
	return (*mExecutors[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].e_ConditionType])(_mEvent, _mFilter, _iCurrentNodeIndex);
}

};

#endif


