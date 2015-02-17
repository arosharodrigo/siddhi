#ifndef _GPU_FILTER_KERNEL_CORE_CU__
#define _GPU_FILTER_KERNEL_CORE_CU__

#include "GpuFilterKernelCore.h"
#include "GpuFilterProcessor.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>
#include <assert.h>

namespace SiddhiGpu
{

// ========================= INT ==============================================

__device__ int AddExpressionInt(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) +
			ExecuteIntExpression(_rParameters));
}

__device__ int MinExpressionInt(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) -
			ExecuteIntExpression(_rParameters));
}

__device__ int MulExpressionInt(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) *
			ExecuteIntExpression(_rParameters));
}

__device__ int DivExpressionInt(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) /
			ExecuteIntExpression(_rParameters));
}

__device__ int ModExpressionInt(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) %
			ExecuteIntExpression(_rParameters));
}

// ========================= LONG ==============================================

__device__ int64_t AddExpressionLong(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) +
			ExecuteLongExpression(_rParameters));
}

__device__ int64_t MinExpressionLong(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) -
			ExecuteLongExpression(_rParameters));
}

__device__ int64_t MulExpressionLong(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) *
			ExecuteLongExpression(_rParameters));
}

__device__ int64_t DivExpressionLong(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) /
			ExecuteLongExpression(_rParameters));
}

__device__ int64_t ModExpressionLong(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) %
			ExecuteLongExpression(_rParameters));
}


// ========================= FLOAT ==============================================

__device__ float AddExpressionFloat(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteFloatExpression(_rParameters) +
			ExecuteFloatExpression(_rParameters));
}

__device__ float MinExpressionFloat(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteFloatExpression(_rParameters) -
			ExecuteFloatExpression(_rParameters));
}

__device__ float MulExpressionFloat(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteFloatExpression(_rParameters) *
			ExecuteFloatExpression(_rParameters));
}

__device__ float DivExpressionFloat(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteFloatExpression(_rParameters) /
			ExecuteFloatExpression(_rParameters));
}

__device__ float ModExpressionFloat(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
//	return ((int64_t)ExecuteFloatExpression(_rParameters) %
//			(int64_t)ExecuteFloatExpression(_rParameters));

	return fmod(ExecuteFloatExpression(_rParameters),
			ExecuteFloatExpression(_rParameters));
}

// ========================= DOUBLE ===========================================

__device__ double AddExpressionDouble(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteDoubleExpression(_rParameters) +
			ExecuteDoubleExpression(_rParameters));
}

__device__ double MinExpressionDouble(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteDoubleExpression(_rParameters) -
			ExecuteDoubleExpression(_rParameters));
}

__device__ double MulExpressionDouble(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteDoubleExpression(_rParameters) *
			ExecuteDoubleExpression(_rParameters));
}

__device__ double DivExpressionDouble(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return (ExecuteDoubleExpression(_rParameters) /
			ExecuteDoubleExpression(_rParameters));
}

__device__ double ModExpressionDouble(FilterEvalParameters & _rParameters)
{
	_rParameters.i_CurrentIndex++;
	return fmod(ExecuteDoubleExpression(_rParameters),
				ExecuteDoubleExpression(_rParameters));
}

/// ============================ CONDITION EXECUTORS ==========================

__device__ bool InvalidOperator(FilterEvalParameters & _rParameters)
{
	return false;
}

__device__ bool NoopOperator(FilterEvalParameters & _rParameters)
{
	return true;
}

// Equal operators

__device__ bool EqualCompareBoolBool(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

__device__ bool EqualCompareIntInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

__device__ bool EqualCompareIntLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) == ExecuteLongExpression(_rParameters));
}

__device__ bool EqualCompareIntFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) == ExecuteFloatExpression(_rParameters));
}

__device__ bool EqualCompareIntDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) == ExecuteDoubleExpression(_rParameters));
}

__device__ bool EqualCompareLongInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

__device__ bool EqualCompareLongLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) == ExecuteLongExpression(_rParameters));
}

__device__ bool EqualCompareLongFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) == ExecuteFloatExpression(_rParameters));
}

__device__ bool EqualCompareLongDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) == ExecuteDoubleExpression(_rParameters));
}

__device__ bool EqualCompareFloatInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

__device__ bool EqualCompareFloatLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) == ExecuteLongExpression(_rParameters));
}

__device__ bool EqualCompareFloatFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) == ExecuteFloatExpression(_rParameters));
}

__device__ bool EqualCompareFloatDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) == ExecuteDoubleExpression(_rParameters));
}

__device__ bool EqualCompareDoubleInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

__device__ bool EqualCompareDoubleLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) == ExecuteLongExpression(_rParameters));
}

__device__ bool EqualCompareDoubleFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) == ExecuteFloatExpression(_rParameters));
}

__device__ bool EqualCompareDoubleDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) == ExecuteDoubleExpression(_rParameters));
}

__device__ bool EqualCompareStringString(FilterEvalParameters & _rParameters)
{
	return (cuda_strcmp(ExecuteStringExpression(_rParameters), ExecuteStringExpression(_rParameters)));
}

/// ============================================================================
// NotEqual operator

__device__ bool NotEqualCompareBoolBool(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

__device__ bool NotEqualCompareIntInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

__device__ bool NotEqualCompareIntLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) != ExecuteLongExpression(_rParameters));
}

__device__ bool NotEqualCompareIntFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) != ExecuteFloatExpression(_rParameters));
}

__device__ bool NotEqualCompareIntDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) != ExecuteDoubleExpression(_rParameters));
}

__device__ bool NotEqualCompareLongInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

__device__ bool NotEqualCompareLongLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) != ExecuteLongExpression(_rParameters));
}

__device__ bool NotEqualCompareLongFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) != ExecuteFloatExpression(_rParameters));
}

__device__ bool NotEqualCompareLongDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) != ExecuteDoubleExpression(_rParameters));
}

__device__ bool NotEqualCompareFloatInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

__device__ bool NotEqualCompareFloatLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) != ExecuteLongExpression(_rParameters));
}

__device__ bool NotEqualCompareFloatFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) != ExecuteFloatExpression(_rParameters));
}

__device__ bool NotEqualCompareFloatDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) != ExecuteDoubleExpression(_rParameters));
}

__device__ bool NotEqualCompareDoubleInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

__device__ bool NotEqualCompareDoubleLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) != ExecuteLongExpression(_rParameters));
}

__device__ bool NotEqualCompareDoubleFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) != ExecuteFloatExpression(_rParameters));
}

__device__ bool NotEqualCompareDoubleDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) != ExecuteDoubleExpression(_rParameters));
}

__device__ bool NotEqualCompareStringString(FilterEvalParameters & _rParameters)
{
	return (!cuda_strcmp(ExecuteStringExpression(_rParameters),ExecuteStringExpression(_rParameters)));
}

/// ============================================================================

// GreaterThan operator

__device__ bool GreaterThanCompareIntInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) > ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterThanCompareIntLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) > ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterThanCompareIntFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) > ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterThanCompareIntDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) > ExecuteDoubleExpression(_rParameters));
}

__device__ bool GreaterThanCompareLongInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) > ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterThanCompareLongLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) > ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterThanCompareLongFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) > ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterThanCompareLongDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) > ExecuteDoubleExpression(_rParameters));
}

__device__ bool GreaterThanCompareFloatInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) > ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterThanCompareFloatLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) > ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterThanCompareFloatFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) > ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterThanCompareFloatDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) > ExecuteDoubleExpression(_rParameters));
}

__device__ bool GreaterThanCompareDoubleInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) > ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterThanCompareDoubleLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) > ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterThanCompareDoubleFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) > ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterThanCompareDoubleDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) > ExecuteDoubleExpression(_rParameters));
}


/// ============================================================================
// LessThan operator

__device__ bool LessThanCompareIntInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) < ExecuteIntExpression(_rParameters));
}

__device__ bool LessThanCompareIntLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) < ExecuteLongExpression(_rParameters));
}

__device__ bool LessThanCompareIntFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) < ExecuteFloatExpression(_rParameters));
}

__device__ bool LessThanCompareIntDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) < ExecuteDoubleExpression(_rParameters));
}

__device__ bool LessThanCompareLongInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) < ExecuteIntExpression(_rParameters));
}

__device__ bool LessThanCompareLongLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) < ExecuteLongExpression(_rParameters));
}

__device__ bool LessThanCompareLongFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) < ExecuteFloatExpression(_rParameters));
}

__device__ bool LessThanCompareLongDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) < ExecuteDoubleExpression(_rParameters));
}

__device__ bool LessThanCompareFloatInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) < ExecuteIntExpression(_rParameters));
}

__device__ bool LessThanCompareFloatLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) < ExecuteLongExpression(_rParameters));
}

__device__ bool LessThanCompareFloatFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) < ExecuteFloatExpression(_rParameters));
}

__device__ bool LessThanCompareFloatDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) < ExecuteDoubleExpression(_rParameters));
}

__device__ bool LessThanCompareDoubleInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) < ExecuteIntExpression(_rParameters));
}

__device__ bool LessThanCompareDoubleLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) < ExecuteLongExpression(_rParameters));
}

__device__ bool LessThanCompareDoubleFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) < ExecuteFloatExpression(_rParameters));
}

__device__ bool LessThanCompareDoubleDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) < ExecuteDoubleExpression(_rParameters));
}

/// ============================================================================
// GreaterAndEqual operator

__device__ bool GreaterAndEqualCompareIntInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) >= ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareIntLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) >= ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareIntFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) >= ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareIntDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) >= ExecuteDoubleExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareLongInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) >= ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareLongLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) >= ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareLongFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) >= ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareLongDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) >= ExecuteDoubleExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareFloatInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) >= ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareFloatLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) >= ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareFloatFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) >= ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareFloatDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) >= ExecuteDoubleExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareDoubleInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) >= ExecuteIntExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareDoubleLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) >= ExecuteLongExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareDoubleFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) >= ExecuteFloatExpression(_rParameters));
}

__device__ bool GreaterAndEqualCompareDoubleDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) >= ExecuteDoubleExpression(_rParameters));
}

/// ============================================================================
// LessAndEqual operator

__device__ bool LessAndEqualCompareIntInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) <= ExecuteIntExpression(_rParameters));
}

__device__ bool LessAndEqualCompareIntLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) <= ExecuteLongExpression(_rParameters));
}

__device__ bool LessAndEqualCompareIntFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) <= ExecuteFloatExpression(_rParameters));
}

__device__ bool LessAndEqualCompareIntDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteIntExpression(_rParameters) <= ExecuteDoubleExpression(_rParameters));
}

__device__ bool LessAndEqualCompareLongInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) <= ExecuteIntExpression(_rParameters));
}

__device__ bool LessAndEqualCompareLongLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) <= ExecuteLongExpression(_rParameters));
}

__device__ bool LessAndEqualCompareLongFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) <= ExecuteFloatExpression(_rParameters));
}

__device__ bool LessAndEqualCompareLongDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteLongExpression(_rParameters) <= ExecuteDoubleExpression(_rParameters));
}

__device__ bool LessAndEqualCompareFloatInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) <= ExecuteIntExpression(_rParameters));
}

__device__ bool LessAndEqualCompareFloatLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) <= ExecuteLongExpression(_rParameters));
}

__device__ bool LessAndEqualCompareFloatFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) <= ExecuteFloatExpression(_rParameters));
}

__device__ bool LessAndEqualCompareFloatDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteFloatExpression(_rParameters) <= ExecuteDoubleExpression(_rParameters));
}

__device__ bool LessAndEqualCompareDoubleInt(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) <= ExecuteIntExpression(_rParameters));
}

__device__ bool LessAndEqualCompareDoubleLong(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) <= ExecuteLongExpression(_rParameters));
}

__device__ bool LessAndEqualCompareDoubleFloat(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) <= ExecuteFloatExpression(_rParameters));
}

__device__ bool LessAndEqualCompareDoubleDouble(FilterEvalParameters & _rParameters)
{
	return (ExecuteDoubleExpression(_rParameters) <= ExecuteDoubleExpression(_rParameters));
}


__device__ bool ContainsOperator(FilterEvalParameters & _rParameters)
{
	return (cuda_contains(ExecuteStringExpression(_rParameters), ExecuteStringExpression(_rParameters)));
}


/// ============================================================================

__device__ bool ExecuteBoolExpression(FilterEvalParameters & _rParameters)
{
//	assert(_rParameters.i_CurrentIndex < _rParameters.p_Filter->i_NodeCount);

	ExecutorNode & mExecutorNode = _rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

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
			if(mExecutorNode.m_VarValue.e_Type == DataType::Boolean &&
					_rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == DataType::Boolean)
			{
				// get attribute value
				int16_t i;
				memcpy(&i, _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 2);
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

__device__ int ExecuteIntExpression(FilterEvalParameters & _rParameters)
{
//	assert(_rParameters.i_CurrentIndex < _rParameters.p_Filter->i_NodeCount);

	ExecutorNode & mExecutorNode = _rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

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
			if(mExecutorNode.m_VarValue.e_Type == DataType::Int &&
					_rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == DataType::Int)
			{
				int32_t i;
				memcpy(&i, _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 4);
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

__device__ int64_t ExecuteLongExpression(FilterEvalParameters & _rParameters)
{
//	assert(_rParameters.i_CurrentIndex < _rParameters.p_Filter->i_NodeCount);
//	printf("EL=%d.%d|%d\n", blockIdx.x, threadIdx.x, _rParameters.i_CurrentIndex);

	ExecutorNode & mExecutorNode = _rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

	if(mExecutorNode.e_NodeType == EXECUTOR_NODE_EXPRESSION)
	{
		switch(mExecutorNode.e_ExpressionType)
		{
		case EXPRESSION_CONST:
		{
			if(mExecutorNode.m_ConstValue.e_Type == DataType::Long)
			{
//				printf("EC=%d.%d|%d\n", blockIdx.x, threadIdx.x, _rParameters.i_CurrentIndex);
				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.l_LongVal;
			}
		}
		break;
		case EXPRESSION_VARIABLE:
		{
			if(mExecutorNode.m_VarValue.e_Type == DataType::Long &&
					_rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == DataType::Long)
			{
				int64_t i;
				memcpy(&i, _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 8);
//				printf("EV=%d.%d|%d\n", blockIdx.x, threadIdx.x, _rParameters.i_CurrentIndex);
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

__device__ float ExecuteFloatExpression(FilterEvalParameters & _rParameters)
{
//	assert(_rParameters.i_CurrentIndex < _rParameters.p_Filter->i_NodeCount);

	ExecutorNode & mExecutorNode = _rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

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
			if(mExecutorNode.m_VarValue.e_Type == DataType::Float &&
					_rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == DataType::Float)
			{
				float f;
				memcpy(&f, _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 4);
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

__device__ double ExecuteDoubleExpression(FilterEvalParameters & _rParameters)
{
//	assert(_rParameters.i_CurrentIndex < _rParameters.p_Filter->i_NodeCount);

	ExecutorNode & mExecutorNode = _rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

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
			if(mExecutorNode.m_VarValue.e_Type == DataType::Double &&
					_rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == DataType::Double)
			{
				double f;
				memcpy(&f, _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 8);
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

__device__ const char * ExecuteStringExpression(FilterEvalParameters & _rParameters)
{
//	assert(_rParameters.i_CurrentIndex < _rParameters.p_Filter->i_NodeCount);

	ExecutorNode & mExecutorNode = _rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

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
			if(mExecutorNode.m_VarValue.e_Type == DataType::StringIn &&
					_rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == DataType::StringIn)
			{
				int16_t i;
				memcpy(&i, _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 2);
				char * z = _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position + 2;
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

		InvalidOperator
};


// ==================================================================================================

__device__ bool AndCondition(FilterEvalParameters & _rParameters)
{
//	return (Evaluate(_rParameters) & Evaluate(_rParameters));
	return (*mExecutors[_rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters) &
			(*mExecutors[_rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters);
}

__device__ bool OrCondition(FilterEvalParameters & _rParameters)
{
	//return (Evaluate(_rParameters) | Evaluate(_rParameters));
	return (*mExecutors[_rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters) |
			(*mExecutors[_rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters);
}

__device__ bool NotCondition(FilterEvalParameters & _rParameters)
{
	//return (!Evaluate(_rParameters));
	return !((*mExecutors[_rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters));
}

__device__ bool BooleanCondition(FilterEvalParameters & _rParameters)
{
	//return (Evaluate(_rParameters));
	return (*mExecutors[_rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters);
}


// =========================================

// evaluate event with an executor tree
__device__ bool Evaluate(FilterEvalParameters & _rParameters)
{
//	assert(_rParameters.i_CurrentIndex < _rParameters.p_Filter->i_NodeCount);
//	assert(_rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex].e_ConditionType < EXECUTOR_CONDITION_COUNT);

	//printf("E=%d.%d|%d\n",blockIdx.x, threadIdx.x,_rParameters.i_CurrentIndex);

	return (*mExecutors[_rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters);
}

};

#endif
