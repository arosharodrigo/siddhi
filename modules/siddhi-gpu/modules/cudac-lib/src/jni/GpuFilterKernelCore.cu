#ifndef _GPU_FILTER_KERNEL_CORE_CU__
#define _GPU_FILTER_KERNEL_CORE_CU__

#include "GpuFilterKernelCore.h"
#include <stdio.h>
#include <stdlib.h>
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

__device__ int AddExpressionInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int MinExpressionInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int MulExpressionInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int DivExpressionInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int ModExpressionInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) %
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

// ========================= LONG ==============================================

__device__ int64_t AddExpressionLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int64_t MinExpressionLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int64_t MulExpressionLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int64_t DivExpressionLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ int64_t ModExpressionLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) %
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}


// ========================= FLOAT ==============================================

__device__ float AddExpressionFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ float MinExpressionFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ float MulExpressionFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ float DivExpressionFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ float ModExpressionFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
//	return ((int64_t)ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) %
//			(int64_t)ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));

	return fmod(ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex),
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

// ========================= DOUBLE ===========================================

__device__ double AddExpressionDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) +
			ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ double MinExpressionDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) -
			ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ double MulExpressionDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) *
			ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ double DivExpressionDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) /
			ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ double ModExpressionDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
//	return ((int64_t)ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) %
//			(int64_t)ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));

	return fmod(ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex),
				ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

/// ============================ CONDITION EXECUTORS ==========================

__device__ bool InvalidOperator(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return false;
}

__device__ bool NoopOperator(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return true;
}

// Equal operators

__device__ bool EqualCompareBoolBool(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareIntInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareIntLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareIntFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareIntDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareLongInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareLongLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareLongFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareLongDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareStringString(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (cuda_strcmp(ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex), ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex)));
}

//__device__ bool EqualCompareExecutorExecutor(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
////	return (Execute(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex) ==
////			Execute(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex));
//}

/// ============================================================================
// NotEqual operator

__device__ bool NotEqualCompareBoolBool(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareStringString(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (!cuda_strcmp(ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex),ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex)));
}

/// ============================================================================

// GreaterThan operator

__device__ bool GreaterThanCompareIntInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareIntLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareIntFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareIntDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}


/// ============================================================================
// LessThan operator

__device__ bool LessThanCompareIntInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareIntLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareIntFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareIntDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

/// ============================================================================
// GreaterAndEqual operator

__device__ bool GreaterAndEqualCompareIntInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareIntLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareIntFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareIntDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

/// ============================================================================
// LessAndEqual operator

__device__ bool LessAndEqualCompareIntInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareIntLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareIntFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareIntDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}


__device__ bool ContainsOperator(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (cuda_contains(ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex), ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex)));
}


/// ============================================================================

__device__ bool ExecuteBoolExpression(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
					_pEventMeta->p_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Boolean)
			{
				// get attribute value
				int16_t i;
				memcpy(&i, _pEvent + _pEventMeta->p_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 2);
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

__device__ int ExecuteIntExpression(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
					_pEventMeta->p_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Int)
			{
				int32_t i;
				memcpy(&i, _pEvent + _pEventMeta->p_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);
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

__device__ int64_t ExecuteLongExpression(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
					_pEventMeta->p_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Long)
			{
				int64_t i;
				memcpy(&i, _pEvent + _pEventMeta->p_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);
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

__device__ float ExecuteFloatExpression(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
					_pEventMeta->p_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Float)
			{
				float f;
				memcpy(&f, _pEvent + _pEventMeta->p_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);
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

__device__ double ExecuteDoubleExpression(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
					_pEventMeta->p_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Double)
			{
				double f;
				memcpy(&f, _pEvent + _pEventMeta->p_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);
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

__device__ const char * ExecuteStringExpression(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
					_pEventMeta->p_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::StringIn)
			{
				int16_t i;
				memcpy(&i, _pEvent + _pEventMeta->p_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 2);
				char * z = _pEvent + _pEventMeta->p_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position + 2;
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

__device__ bool AndCondition(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) && Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool OrCondition(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex) || Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool NotCondition(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (!Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex));
}

__device__ bool BooleanCondition(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
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
__device__ bool Evaluate(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex)
{
	return (*mExecutors[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].e_ConditionType])(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex);
}

};

#endif
