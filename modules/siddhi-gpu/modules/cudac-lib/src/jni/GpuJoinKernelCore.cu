#ifndef _GPU_JOIN_KERNEL_CORE_CU__
#define _GPU_JOIN_KERNEL_CORE_CU__

#include "GpuJoinKernelCore.h"
#include "GpuFilterProcessor.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

namespace SiddhiGpu
{

__device__ int AddExpressionInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) +
			ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ int MinExpressionInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) -
			ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ int MulExpressionInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) *
			ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ int DivExpressionInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) /
			ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ int ModExpressionInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) %
			ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

// ========================= LONG ==============================================

__device__ int64_t AddExpressionLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) +
			ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ int64_t MinExpressionLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) -
			ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ int64_t MulExpressionLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) *
			ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ int64_t DivExpressionLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) /
			ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ int64_t ModExpressionLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) %
			ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}


// ========================= FLOAT ==============================================

__device__ float AddExpressionFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) +
			ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ float MinExpressionFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) -
			ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ float MulExpressionFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) *
			ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ float DivExpressionFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) /
			ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ float ModExpressionFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
//	return ((int64_t)ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) %
//			(int64_t)ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));

	return fmod(ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex),
			ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

// ========================= DOUBLE ===========================================

__device__ double AddExpressionDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) +
			ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ double MinExpressionDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) -
			ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ double MulExpressionDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) *
			ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ double DivExpressionDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) /
			ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ double ModExpressionDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	_iCurrentNodeIndex++;
//	return ((int64_t)ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) %
//			(int64_t)ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));

	return fmod(ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex),
				ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

/// ============================ CONDITION EXECUTORS ==========================

__device__ bool InvalidOperator(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return false;
}

__device__ bool NoopOperator(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return true;
}

// Equal operators

__device__ bool EqualCompareBoolBool(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareIntInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareIntLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareIntFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareIntDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareLongInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareLongLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareLongFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareLongDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareFloatDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareDoubleDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) == ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool EqualCompareStringString(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (cuda_strcmp(ExecuteStringExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex), ExecuteStringExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex)));
}

//__device__ bool EqualCompareExecutorExecutor(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
//{
//	switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex + 1].e_NodeType)
//	{
//		case EXECUTOR_NODE_CONST:
//		{
//			switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex + 1].m_ConstValue.e_Type)
//			{
//			case INT:
//			{
////				return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex++) ==
////						ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex++));
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
////			return ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex++);
//		default:
//			break;
//	}
////	return (Execute(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex) ==
////			Execute(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex));
//}

/// ============================================================================
// NotEqual operator

__device__ bool NotEqualCompareBoolBool(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareIntDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareLongDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareFloatDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareDoubleDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) != ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotEqualCompareStringString(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (!cuda_strcmp(ExecuteStringExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex),ExecuteStringExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex)));
}

/// ============================================================================

// GreaterThan operator

__device__ bool GreaterThanCompareIntInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareIntLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareIntFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareIntDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareLongDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareFloatDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterThanCompareDoubleDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) > ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}


/// ============================================================================
// LessThan operator

__device__ bool LessThanCompareIntInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareIntLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareIntFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareIntDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareLongDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareFloatDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessThanCompareDoubleDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) < ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

/// ============================================================================
// GreaterAndEqual operator

__device__ bool GreaterAndEqualCompareIntInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareIntLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareIntFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareIntDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareLongDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareFloatDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool GreaterAndEqualCompareDoubleDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) >= ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

/// ============================================================================
// LessAndEqual operator

__device__ bool LessAndEqualCompareIntInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareIntLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareIntFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareIntDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareLongDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareFloatDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteIntExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteLongExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteFloatExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool LessAndEqualCompareDoubleDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) <= ExecuteDoubleExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}


__device__ bool ContainsOperator(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (cuda_contains(ExecuteStringExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex), ExecuteStringExpression(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex)));
}


/// ============================================================================


__device__ bool ExecuteBoolExpression(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::Boolean)
			{
				return _mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.b_BoolVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			// if filter data type matches event attribute data type, return attribute value
			if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_StreamIndex == 0)
			{
				if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Boolean &&
						_pLeftMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Boolean)
				{
					// get attribute value
					int16_t i;
					memcpy(&i, _pLeftEvent + _pLeftMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 2);
					_iCurrentNodeIndex++;
					return i;
				}
			}
			else if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_StreamIndex == 1)
			{
				if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Boolean &&
						_pRightMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Boolean)
				{
					// get attribute value
					int16_t i;
					memcpy(&i, _pRightEvent + _pRightMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 2);
					_iCurrentNodeIndex++;
					return i;
				}
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

__device__ int ExecuteIntExpression(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::Int)
			{
				return _mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.i_IntVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_StreamIndex == 0)
			{
				if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Int &&
						_pLeftMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Int)
				{
					int32_t i;
					memcpy(&i, _pLeftEvent + _pLeftMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);
					_iCurrentNodeIndex++;
					return i;
				}
			}
			else if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_StreamIndex == 1)
			{
				if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Int &&
						_pRightMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Int)
				{
					int32_t i;
					memcpy(&i, _pRightEvent + _pRightMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);
					_iCurrentNodeIndex++;
					return i;
				}
			}
		}
		break;
		case EXPRESSION_ADD_INT:
		{
			return AddExpressionInt(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_INT:
		{
			return MinExpressionInt(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_INT:
		{
			return MulExpressionInt(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_INT:
		{
			return DivExpressionInt(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_INT:
		{
			return ModExpressionInt(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return INT_MIN;
}

__device__ int64_t ExecuteLongExpression(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::Long)
			{
				return _mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.l_LongVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_StreamIndex == 0)
			{
				if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Long &&
						_pLeftMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Long)
				{
					int64_t i;
					memcpy(&i, _pLeftEvent + _pLeftMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);
					_iCurrentNodeIndex++;
					return i;
				}
			}
			else if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_StreamIndex == 1)
			{
				if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Long &&
						_pRightMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Long)
				{
					int64_t i;
					memcpy(&i, _pRightEvent + _pRightMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);
					_iCurrentNodeIndex++;
					return i;
				}
			}
		}
		break;
		case EXPRESSION_ADD_LONG:
		{
			return AddExpressionLong(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_LONG:
		{
			return MinExpressionLong(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_LONG:
		{
			return MulExpressionLong(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_LONG:
		{
			return DivExpressionLong(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_LONG:
		{
			return ModExpressionLong(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return LLONG_MIN;
}

__device__ float ExecuteFloatExpression(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::Float)
			{
				return _mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.f_FloatVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_StreamIndex == 0)
			{
				if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Float &&
						_pLeftMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Float)
				{
					float f;
					memcpy(&f, _pLeftEvent + _pLeftMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);
					_iCurrentNodeIndex++;
					return f;
				}
			}
			else if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_StreamIndex == 1)
			{
				if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Float &&
						_pRightMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Float)
				{
					float f;
					memcpy(&f, _pRightEvent + _pRightMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);
					_iCurrentNodeIndex++;
					return f;
				}
			}
		}
		break;
		case EXPRESSION_ADD_FLOAT:
		{
			return AddExpressionFloat(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_FLOAT:
		{
			return MinExpressionFloat(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_FLOAT:
		{
			return MulExpressionFloat(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_FLOAT:
		{
			return DivExpressionFloat(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_FLOAT:
		{
			return ModExpressionFloat(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return FLT_MIN;
}

__device__ double ExecuteDoubleExpression(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::Double)
			{
				return _mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.d_DoubleVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_StreamIndex == 0)
			{
				if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Double &&
						_pLeftMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Double)
				{
					double f;
					memcpy(&f, _pLeftEvent + _pLeftMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);
					_iCurrentNodeIndex++;
					return f;
				}
			}
			else if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_StreamIndex == 1)
			{
				if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::Double &&
						_pRightMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::Double)
				{
					double f;
					memcpy(&f, _pRightEvent + _pRightMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);
					_iCurrentNodeIndex++;
					return f;
				}
			}
		}
		break;
		case EXPRESSION_ADD_DOUBLE:
		{
			return AddExpressionDouble(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_SUB_DOUBLE:
		{
			return MinExpressionDouble(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MUL_DOUBLE:
		{
			return MulExpressionDouble(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_DIV_DOUBLE:
		{
			return DivExpressionDouble(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		case EXPRESSION_MOD_DOUBLE:
		{
			return ModExpressionDouble(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
		}
		default:
			break;
		}
	}

	_iCurrentNodeIndex++;
	return DBL_MIN;
}

__device__ const char * ExecuteStringExpression(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::StringIn)
			{
				return _mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.z_StringVal;
			}
			else if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == DataType::StringExt)
			{
				return _mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.z_ExtString;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_StreamIndex == 0)
			{
				if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::StringIn &&
						_pLeftMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::StringIn)
				{
					int16_t i;
					memcpy(&i, _pLeftEvent + _pLeftMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 2);
					char * z = _pLeftEvent + _pLeftMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position + 2;
					z[i] = 0;
					_iCurrentNodeIndex++;
					return z;
				}
			}
			else if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_StreamIndex == 1)
			{
				if(_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == DataType::StringIn &&
						_pRightMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == DataType::StringIn)
				{
					int16_t i;
					memcpy(&i, _pRightEvent + _pRightMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 2);
					char * z = _pRightEvent + _pRightMeta->p_Attributes[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position + 2;
					z[i] = 0;
					_iCurrentNodeIndex++;
					return z;
				}
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

__device__ bool AndCondition(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) && Evaluate(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool OrCondition(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex) || Evaluate(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool NotCondition(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (!Evaluate(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

__device__ bool BooleanCondition(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (Evaluate(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex));
}

// =================================================================================================

// set all the executor functions here
__device__ OnCompareFuncPointer mOnCompareExecutors[EXECUTOR_CONDITION_COUNT] = {
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
__device__ bool Evaluate(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex)
{
	return (*mOnCompareExecutors[_mOnCompare.ap_ExecutorNodes[_iCurrentNodeIndex++].e_ConditionType])(_mOnCompare, _pLeftMeta, _pLeftEvent, _pRightMeta, _pRightEvent, _iCurrentNodeIndex);
}

}

#endif
