#ifndef _GPU_JOIN_KERNEL_CORE_CU__
#define _GPU_JOIN_KERNEL_CORE_CU__

#include "GpuJoinKernelCore.h"
#include "GpuFilterProcessor.h"
#include "GpuJoinProcessor.h"
#include "GpuCudaHelper.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>
#include <assert.h>

namespace SiddhiGpu
{

__device__ int AddExpressionInt(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) +
			ExecuteIntExpression(_rParameters));
}

__device__ int MinExpressionInt(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) -
			ExecuteIntExpression(_rParameters));
}

__device__ int MulExpressionInt(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) *
			ExecuteIntExpression(_rParameters));
}

__device__ int DivExpressionInt(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) /
			ExecuteIntExpression(_rParameters));
}

__device__ int ModExpressionInt(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) %
			ExecuteIntExpression(_rParameters));
}

// ========================= LONG ==============================================

__device__ int64_t AddExpressionLong(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) +
			ExecuteLongExpression(_rParameters));
}

__device__ int64_t MinExpressionLong(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) -
			ExecuteLongExpression(_rParameters));
}

__device__ int64_t MulExpressionLong(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) *
			ExecuteLongExpression(_rParameters));
}

__device__ int64_t DivExpressionLong(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) /
			ExecuteLongExpression(_rParameters));
}

__device__ int64_t ModExpressionLong(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) %
			ExecuteLongExpression(_rParameters));
}


// ========================= FLOAT ==============================================

__device__ float AddExpressionFloat(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteFloatExpression(_rParameters) +
			ExecuteFloatExpression(_rParameters));
}

__device__ float MinExpressionFloat(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteFloatExpression(_rParameters) -
			ExecuteFloatExpression(_rParameters));
}

__device__ float MulExpressionFloat(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteFloatExpression(_rParameters) *
			ExecuteFloatExpression(_rParameters));
}

__device__ float DivExpressionFloat(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteFloatExpression(_rParameters) /
			ExecuteFloatExpression(_rParameters));
}

__device__ float ModExpressionFloat(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
//	return ((int64_t)ExecuteFloatExpression(_rParameters) %
//			(int64_t)ExecuteFloatExpression(_rParameters));

	return fmod(ExecuteFloatExpression(_rParameters),
			ExecuteFloatExpression(_rParameters));
}

// ========================= DOUBLE ===========================================

__device__ double AddExpressionDouble(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteDoubleExpression(_rParameters) +
			ExecuteDoubleExpression(_rParameters));
}

__device__ double MinExpressionDouble(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteDoubleExpression(_rParameters) -
			ExecuteDoubleExpression(_rParameters));
}

__device__ double MulExpressionDouble(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteDoubleExpression(_rParameters) *
			ExecuteDoubleExpression(_rParameters));
}

__device__ double DivExpressionDouble(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteDoubleExpression(_rParameters) /
			ExecuteDoubleExpression(_rParameters));
}

__device__ double ModExpressionDouble(ExpressionEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
//	return ((int64_t)ExecuteDoubleExpression(_rParameters) %
//			(int64_t)ExecuteDoubleExpression(_rParameters));

	return fmod(ExecuteDoubleExpression(_rParameters),
				ExecuteDoubleExpression(_rParameters));
}

/// ============================ CONDITION EXECUTORS ==========================

__device__ bool InvalidOperator(ExpressionEvalParameters & /*_rParameters*/)
{
	return false;
}

__device__ bool NoopOperator(ExpressionEvalParameters & /*_rParameters*/)
{
	return true;
}

// Equal operators

__device__ bool EqualCompareBoolBool(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

__device__ bool EqualCompareIntInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

__device__ bool EqualCompareIntLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) == ExecuteLongExpression(_rParameters));
}

__device__ bool EqualCompareIntFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) == ExecuteFloatExpression(_rParameters));
}

__device__ bool EqualCompareIntDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) == ExecuteDoubleExpression(_rParameters));
}

__device__ bool EqualCompareLongInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

__device__ bool EqualCompareLongLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) == ExecuteLongExpression(_rParameters));
}

__device__ bool EqualCompareLongFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) == ExecuteFloatExpression(_rParameters));
}

__device__ bool EqualCompareLongDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) == ExecuteDoubleExpression(_rParameters));
}

__device__ bool EqualCompareFloatInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

__device__ bool EqualCompareFloatLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) == ExecuteLongExpression(_rParameters));
}

__device__ bool EqualCompareFloatFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) == ExecuteFloatExpression(_rParameters));
}

__device__ bool EqualCompareFloatDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) == ExecuteDoubleExpression(_rParameters));
}

__device__ bool EqualCompareDoubleInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

__device__ bool EqualCompareDoubleLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) == ExecuteLongExpression(_rParameters));
}

__device__ bool EqualCompareDoubleFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) == ExecuteFloatExpression(_rParameters));
}

__device__ bool EqualCompareDoubleDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) == ExecuteDoubleExpression(_rParameters));
}

__device__ bool EqualCompareStringString(ExpressionEvalParameters & _rParameters)
{
	return (cuda_strcmp(ExecuteStringExpression(_rParameters), ExecuteStringExpression(_rParameters)));
}

/// ============================================================================
// NotEqual operator

__device__ bool NotEqualCompareBoolBool(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

__device__ bool NotEqualCompareIntInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

__device__ bool NotEqualCompareIntLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) != ExecuteLongExpression(_rParameters));
}

__device__ bool NotEqualCompareIntFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) != ExecuteFloatExpression(_rParameters));
}

__device__ bool NotEqualCompareIntDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) != ExecuteDoubleExpression(_rParameters));
}

__device__ bool NotEqualCompareLongInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

__device__ bool NotEqualCompareLongLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) != ExecuteLongExpression(_rParameters));
}

__device__ bool NotEqualCompareLongFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) != ExecuteFloatExpression(_rParameters));
}

__device__ bool NotEqualCompareLongDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) != ExecuteDoubleExpression(_rParameters));
}

__device__ bool NotEqualCompareFloatInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

__device__ bool NotEqualCompareFloatLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) != ExecuteLongExpression(_rParameters));
}

__device__ bool NotEqualCompareFloatFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) != ExecuteFloatExpression(_rParameters));
}

__device__ bool NotEqualCompareFloatDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) != ExecuteDoubleExpression(_rParameters));
}

__device__ bool NotEqualCompareDoubleInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

__device__ bool NotEqualCompareDoubleLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) != ExecuteLongExpression(_rParameters));
}

__device__ bool NotEqualCompareDoubleFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) != ExecuteFloatExpression(_rParameters));
}

__device__ bool NotEqualCompareDoubleDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) != ExecuteDoubleExpression(_rParameters));
}

__device__ bool NotEqualCompareStringString(ExpressionEvalParameters & _rParameters)
{
	return (!cuda_strcmp(ExecuteStringExpression(_rParameters),ExecuteStringExpression(_rParameters)));
}

/// ============================================================================

// GreaterThan operator

__device__ bool GreaterThanCompareIntInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) > ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterThanCompareIntLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) > ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterThanCompareIntFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) > ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterThanCompareIntDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) > ExecuteDoubleExpression(_rParameters));
}

__device__ bool GreaterThanCompareLongInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) > ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterThanCompareLongLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) > ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterThanCompareLongFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) > ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterThanCompareLongDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) > ExecuteDoubleExpression(_rParameters));
}

__device__ bool GreaterThanCompareFloatInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) > ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterThanCompareFloatLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) > ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterThanCompareFloatFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) > ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterThanCompareFloatDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) > ExecuteDoubleExpression(_rParameters));
}

__device__ bool GreaterThanCompareDoubleInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) > ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterThanCompareDoubleLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) > ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterThanCompareDoubleFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) > ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterThanCompareDoubleDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) > ExecuteDoubleExpression(_rParameters));
}


/// ============================================================================
// LessThan operator

__device__ bool LessThanCompareIntInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) < ExecuteIntExpression(_rParameters));
}

__device__ bool LessThanCompareIntLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) < ExecuteLongExpression(_rParameters));
}

__device__ bool LessThanCompareIntFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) < ExecuteFloatExpression(_rParameters));
}

__device__ bool LessThanCompareIntDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) < ExecuteDoubleExpression(_rParameters));
}

__device__ bool LessThanCompareLongInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) < ExecuteIntExpression(_rParameters));
}

__device__ bool LessThanCompareLongLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) < ExecuteLongExpression(_rParameters));
}

__device__ bool LessThanCompareLongFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) < ExecuteFloatExpression(_rParameters));
}

__device__ bool LessThanCompareLongDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) < ExecuteDoubleExpression(_rParameters));
}

__device__ bool LessThanCompareFloatInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) < ExecuteIntExpression(_rParameters));
}

__device__ bool LessThanCompareFloatLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) < ExecuteLongExpression(_rParameters));
}

__device__ bool LessThanCompareFloatFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) < ExecuteFloatExpression(_rParameters));
}

__device__ bool LessThanCompareFloatDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) < ExecuteDoubleExpression(_rParameters));
}

__device__ bool LessThanCompareDoubleInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) < ExecuteIntExpression(_rParameters));
}

__device__ bool LessThanCompareDoubleLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) < ExecuteLongExpression(_rParameters));
}

__device__ bool LessThanCompareDoubleFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) < ExecuteFloatExpression(_rParameters));
}

__device__ bool LessThanCompareDoubleDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) < ExecuteDoubleExpression(_rParameters));
}

/// ============================================================================
// GreaterAndEqual operator

__device__ bool GreaterAndEqualCompareIntInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) >= ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareIntLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) >= ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareIntFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) >= ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareIntDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) >= ExecuteDoubleExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareLongInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) >= ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareLongLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) >= ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareLongFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) >= ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareLongDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) >= ExecuteDoubleExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareFloatInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) >= ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareFloatLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) >= ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareFloatFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) >= ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareFloatDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) >= ExecuteDoubleExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareDoubleInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) >= ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareDoubleLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) >= ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareDoubleFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) >= ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareDoubleDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) >= ExecuteDoubleExpression(_rParameters));
}

/// ============================================================================
// LessAndEqual operator

__device__ bool LessAndEqualCompareIntInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) <= ExecuteIntExpression(_rParameters));
}

__device__ bool LessAndEqualCompareIntLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) <= ExecuteLongExpression(_rParameters));
}

__device__ bool LessAndEqualCompareIntFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) <= ExecuteFloatExpression(_rParameters));
}

__device__ bool LessAndEqualCompareIntDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) <= ExecuteDoubleExpression(_rParameters));
}

__device__ bool LessAndEqualCompareLongInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) <= ExecuteIntExpression(_rParameters));
}

__device__ bool LessAndEqualCompareLongLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) <= ExecuteLongExpression(_rParameters));
}

__device__ bool LessAndEqualCompareLongFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) <= ExecuteFloatExpression(_rParameters));
}

__device__ bool LessAndEqualCompareLongDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) <= ExecuteDoubleExpression(_rParameters));
}

__device__ bool LessAndEqualCompareFloatInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) <= ExecuteIntExpression(_rParameters));
}

__device__ bool LessAndEqualCompareFloatLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) <= ExecuteLongExpression(_rParameters));
}

__device__ bool LessAndEqualCompareFloatFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) <= ExecuteFloatExpression(_rParameters));
}

__device__ bool LessAndEqualCompareFloatDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) <= ExecuteDoubleExpression(_rParameters));
}

__device__ bool LessAndEqualCompareDoubleInt(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) <= ExecuteIntExpression(_rParameters));
}

__device__ bool LessAndEqualCompareDoubleLong(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) <= ExecuteLongExpression(_rParameters));
}

__device__ bool LessAndEqualCompareDoubleFloat(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) <= ExecuteFloatExpression(_rParameters));
}

__device__ bool LessAndEqualCompareDoubleDouble(ExpressionEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) <= ExecuteDoubleExpression(_rParameters));
}


__device__ bool ContainsOperator(ExpressionEvalParameters & _rParameters)
{
	return (cuda_contains(ExecuteStringExpression(_rParameters), ExecuteStringExpression(_rParameters)));
}


/// ============================================================================


__device__ bool ExecuteBoolExpression(ExpressionEvalParameters & _rParameters)
{
	ExecutorNode & mExecutorNode = _rParameters.p_OnCompare->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

	if(mExecutorNode.e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(mExecutorNode.e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(mExecutorNode.m_ConstValue.e_Type == DataType::Boolean)
			{
				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.b_BoolVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			// if filter data type matches event attribute data type, return attribute value
			int iStreamIndex = mExecutorNode.m_VarValue.i_StreamIndex;

			if(mExecutorNode.m_VarValue.e_Type == DataType::Boolean &&
					_rParameters.a_Meta[iStreamIndex]->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == DataType::Boolean)
			{
				// get attribute value
				int16_t i;
				memcpy(&i, _rParameters.a_Event[iStreamIndex] + _rParameters.a_Meta[iStreamIndex]->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 2);
				_rParameters.i_CurrentIndex++;
				return i;
			}
		}
		break;
		default:
			break;
		}
	}

	_rParameters.i_CurrentIndex++;
	return false;
}

__device__ int ExecuteIntExpression(ExpressionEvalParameters & _rParameters)
{
	ExecutorNode & mExecutorNode = _rParameters.p_OnCompare->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

	if(mExecutorNode.e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(mExecutorNode.e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(mExecutorNode.m_ConstValue.e_Type == DataType::Int)
			{
				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.i_IntVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			int iStreamIndex = mExecutorNode.m_VarValue.i_StreamIndex;

			if(mExecutorNode.m_VarValue.e_Type == DataType::Int &&
					_rParameters.a_Meta[iStreamIndex]->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == DataType::Int)
			{
				int32_t i;
				memcpy(&i, _rParameters.a_Event[iStreamIndex] + _rParameters.a_Meta[iStreamIndex]->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 4);
				_rParameters.i_CurrentIndex++;
				return i;
			}

		}
		break;
		case EXPRESSION_ADD_INT:
		{
			return AddExpressionInt(_rParameters);
		}
		case EXPRESSION_SUB_INT:
		{
			return MinExpressionInt(_rParameters);
		}
		case EXPRESSION_MUL_INT:
		{
			return MulExpressionInt(_rParameters);
		}
		case EXPRESSION_DIV_INT:
		{
			return DivExpressionInt(_rParameters);
		}
		case EXPRESSION_MOD_INT:
		{
			return ModExpressionInt(_rParameters);
		}
		default:
			break;
		}
	}

	_rParameters.i_CurrentIndex++;
	return INT_MIN;
}

__device__ int64_t ExecuteLongExpression(ExpressionEvalParameters & _rParameters)
{
	ExecutorNode & mExecutorNode = _rParameters.p_OnCompare->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

	if(mExecutorNode.e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(mExecutorNode.e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(mExecutorNode.m_ConstValue.e_Type == DataType::Long)
			{
				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.l_LongVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			int iStreamIndex = mExecutorNode.m_VarValue.i_StreamIndex;

			if(mExecutorNode.m_VarValue.e_Type == DataType::Long &&
					_rParameters.a_Meta[iStreamIndex]->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == DataType::Long)
			{
				int64_t i;
				memcpy(&i, _rParameters.a_Event[iStreamIndex] + _rParameters.a_Meta[iStreamIndex]->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 8);
				_rParameters.i_CurrentIndex++;
				return i;
			}

		}
		break;
		case EXPRESSION_ADD_LONG:
		{
			return AddExpressionLong(_rParameters);
		}
		case EXPRESSION_SUB_LONG:
		{
			return MinExpressionLong(_rParameters);
		}
		case EXPRESSION_MUL_LONG:
		{
			return MulExpressionLong(_rParameters);
		}
		case EXPRESSION_DIV_LONG:
		{
			return DivExpressionLong(_rParameters);
		}
		case EXPRESSION_MOD_LONG:
		{
			return ModExpressionLong(_rParameters);
		}
		default:
			break;
		}
	}

	_rParameters.i_CurrentIndex++;
	return LLONG_MIN;
}

__device__ float ExecuteFloatExpression(ExpressionEvalParameters & _rParameters)
{
	ExecutorNode & mExecutorNode = _rParameters.p_OnCompare->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

	if(mExecutorNode.e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(mExecutorNode.e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(mExecutorNode.m_ConstValue.e_Type == DataType::Float)
			{
				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.f_FloatVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			int iStreamIndex = mExecutorNode.m_VarValue.i_StreamIndex;

			if(mExecutorNode.m_VarValue.e_Type == DataType::Float &&
					_rParameters.a_Meta[iStreamIndex]->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == DataType::Float)
			{
				float f;
				memcpy(&f, _rParameters.a_Event[iStreamIndex] + _rParameters.a_Meta[iStreamIndex]->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 4);
				_rParameters.i_CurrentIndex++;
				return f;
			}

		}
		break;
		case EXPRESSION_ADD_FLOAT:
		{
			return AddExpressionFloat(_rParameters);
		}
		case EXPRESSION_SUB_FLOAT:
		{
			return MinExpressionFloat(_rParameters);
		}
		case EXPRESSION_MUL_FLOAT:
		{
			return MulExpressionFloat(_rParameters);
		}
		case EXPRESSION_DIV_FLOAT:
		{
			return DivExpressionFloat(_rParameters);
		}
		case EXPRESSION_MOD_FLOAT:
		{
			return ModExpressionFloat(_rParameters);
		}
		default:
			break;
		}
	}

	_rParameters.i_CurrentIndex++;
	return FLT_MIN;
}

__device__ double ExecuteDoubleExpression(ExpressionEvalParameters & _rParameters)
{
	ExecutorNode & mExecutorNode = _rParameters.p_OnCompare->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

	if(mExecutorNode.e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(mExecutorNode.e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(mExecutorNode.m_ConstValue.e_Type == DataType::Double)
			{
				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.d_DoubleVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			int iStreamIndex = mExecutorNode.m_VarValue.i_StreamIndex;

			if(mExecutorNode.m_VarValue.e_Type == DataType::Double &&
					_rParameters.a_Meta[iStreamIndex]->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == DataType::Double)
			{
				double f;
				memcpy(&f, _rParameters.a_Event[iStreamIndex] + _rParameters.a_Meta[iStreamIndex]->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 8);
				_rParameters.i_CurrentIndex++;
				return f;
			}
		}
		break;
		case EXPRESSION_ADD_DOUBLE:
		{
			return AddExpressionDouble(_rParameters);
		}
		case EXPRESSION_SUB_DOUBLE:
		{
			return MinExpressionDouble(_rParameters);
		}
		case EXPRESSION_MUL_DOUBLE:
		{
			return MulExpressionDouble(_rParameters);
		}
		case EXPRESSION_DIV_DOUBLE:
		{
			return DivExpressionDouble(_rParameters);
		}
		case EXPRESSION_MOD_DOUBLE:
		{
			return ModExpressionDouble(_rParameters);
		}
		default:
			break;
		}
	}

	_rParameters.i_CurrentIndex++;
	return DBL_MIN;
}

__device__ const char * ExecuteStringExpression(ExpressionEvalParameters & _rParameters)
{
	ExecutorNode & mExecutorNode = _rParameters.p_OnCompare->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

	if(mExecutorNode.e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(mExecutorNode.e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(mExecutorNode.m_ConstValue.e_Type == DataType::StringIn)
			{
				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.z_StringVal;
			}
			else if(mExecutorNode.m_ConstValue.e_Type == DataType::StringExt)
			{
				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.z_ExtString;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			int iStreamIndex = mExecutorNode.m_VarValue.i_StreamIndex;

			if(mExecutorNode.m_VarValue.e_Type == DataType::StringIn &&
					_rParameters.a_Meta[iStreamIndex]->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == DataType::StringIn)
			{
				int16_t i;
				memcpy(&i, _rParameters.a_Event[iStreamIndex] + _rParameters.a_Meta[iStreamIndex]->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 2);
				char * z = _rParameters.a_Event[iStreamIndex] + _rParameters.a_Meta[iStreamIndex]->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position + 2;
				z[i] = 0;
				_rParameters.i_CurrentIndex++;
				return z;
			}
		}
		break;
		default:
			break;
		}
	}

	_rParameters.i_CurrentIndex++;
	return NULL;
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

// ==================================================================================================

__device__ bool AndCondition(ExpressionEvalParameters & _rParameters)
{
//	return (Evaluate(_rParameters) & Evaluate(_rParameters));
	return (*mOnCompareExecutors[_rParameters.p_OnCompare->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters) &
			(*mOnCompareExecutors[_rParameters.p_OnCompare->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters);
}

__device__ bool OrCondition(ExpressionEvalParameters & _rParameters)
{
//	return (Evaluate(_rParameters) | Evaluate(_rParameters));
	return (*mOnCompareExecutors[_rParameters.p_OnCompare->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters) |
			(*mOnCompareExecutors[_rParameters.p_OnCompare->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters);
}

__device__ bool NotCondition(ExpressionEvalParameters & _rParameters)
{
//	return (!Evaluate(_rParameters));
	return !((*mOnCompareExecutors[_rParameters.p_OnCompare->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters));
}

__device__ bool BooleanCondition(ExpressionEvalParameters & _rParameters)
{
//	return (Evaluate(_rParameters));
	return ((*mOnCompareExecutors[_rParameters.p_OnCompare->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters));
}

// =========================================

// evaluate event with an executor tree
__device__ bool Evaluate(ExpressionEvalParameters & _rParameters)
{
	return (*mOnCompareExecutors[_rParameters.p_OnCompare->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters);
}

}

#endif
