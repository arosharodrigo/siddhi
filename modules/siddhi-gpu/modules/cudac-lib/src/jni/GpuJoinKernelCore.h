/*
 * GpuJoinKernelCore.h
 *
 *  Created on: Feb 10, 2015
 *      Author: prabodha
 */

#ifndef GPUJOINKERNELCORE_H_
#define GPUJOINKERNELCORE_H_

#include "GpuKernelDataTypes.h"
#include "GpuCudaCommon.h"

namespace SiddhiGpu
{

class GpuJoinProcessor;

typedef struct ExpressionEvalParameters
{
	GpuKernelFilter      * p_OnCompare;
	GpuKernelMetaEvent   * a_Meta[2];
	char                 * a_Event[2];
	int                    i_CurrentIndex;
} ExpressionEvalParameters;

// executor function pointer type
typedef bool (*OnCompareFuncPointer)(ExpressionEvalParameters &);

extern __device__ bool NoopOperator(ExpressionEvalParameters &);

extern __device__ bool AndCondition(ExpressionEvalParameters &);
extern __device__ bool OrCondition(ExpressionEvalParameters &);
extern __device__ bool NotCondition(ExpressionEvalParameters &);
extern __device__ bool BooleanCondition(ExpressionEvalParameters &);

extern __device__ bool EqualCompareBoolBool(ExpressionEvalParameters &);
extern __device__ bool EqualCompareIntInt(ExpressionEvalParameters &);
extern __device__ bool EqualCompareIntLong(ExpressionEvalParameters &);
extern __device__ bool EqualCompareIntFloat(ExpressionEvalParameters &);
extern __device__ bool EqualCompareIntDouble(ExpressionEvalParameters &);
extern __device__ bool EqualCompareLongInt(ExpressionEvalParameters &);
extern __device__ bool EqualCompareLongLong(ExpressionEvalParameters &);
extern __device__ bool EqualCompareLongFloat(ExpressionEvalParameters &);
extern __device__ bool EqualCompareLongDouble(ExpressionEvalParameters &);
extern __device__ bool EqualCompareFloatInt(ExpressionEvalParameters &);
extern __device__ bool EqualCompareFloatLong(ExpressionEvalParameters &);
extern __device__ bool EqualCompareFloatFloat(ExpressionEvalParameters &);
extern __device__ bool EqualCompareFloatDouble(ExpressionEvalParameters &);
extern __device__ bool EqualCompareDoubleInt(ExpressionEvalParameters &);
extern __device__ bool EqualCompareDoubleLong(ExpressionEvalParameters &);
extern __device__ bool EqualCompareDoubleFloat(ExpressionEvalParameters &);
extern __device__ bool EqualCompareDoubleDouble(ExpressionEvalParameters &);
extern __device__ bool EqualCompareStringString(ExpressionEvalParameters &);
//extern __device__ bool EqualCompareExecutorExecutor(ExpressionEvalParameters &);

extern __device__ bool NotEqualCompareBoolBool(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareIntInt(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareIntLong(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareIntFloat(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareIntDouble(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareLongInt(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareLongLong(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareLongFloat(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareLongDouble(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareFloatInt(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareFloatLong(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareFloatFloat(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareFloatDouble(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareDoubleInt(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareDoubleLong(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareDoubleFloat(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareDoubleDouble(ExpressionEvalParameters &);
extern __device__ bool NotEqualCompareStringString(ExpressionEvalParameters &);

extern __device__ bool GreaterThanCompareIntInt(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareIntLong(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareIntFloat(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareIntDouble(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareLongInt(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareLongLong(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareLongFloat(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareLongDouble(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareFloatInt(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareFloatLong(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareFloatFloat(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareFloatDouble(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareDoubleInt(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareDoubleLong(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareDoubleFloat(ExpressionEvalParameters &);
extern __device__ bool GreaterThanCompareDoubleDouble(ExpressionEvalParameters &);

extern __device__ bool LessThanCompareIntInt(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareIntLong(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareIntFloat(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareIntDouble(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareLongInt(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareLongLong(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareLongFloat(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareLongDouble(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareFloatInt(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareFloatLong(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareFloatFloat(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareFloatDouble(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareDoubleInt(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareDoubleLong(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareDoubleFloat(ExpressionEvalParameters &);
extern __device__ bool LessThanCompareDoubleDouble(ExpressionEvalParameters &);

extern __device__ bool GreaterAndEqualCompareIntInt(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareIntLong(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareIntFloat(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareIntDouble(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareLongInt(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareLongLong(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareLongFloat(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareLongDouble(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareFloatInt(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareFloatLong(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareFloatFloat(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareFloatDouble(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareDoubleInt(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareDoubleLong(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareDoubleFloat(ExpressionEvalParameters &);
extern __device__ bool GreaterAndEqualCompareDoubleDouble(ExpressionEvalParameters &);

extern __device__ bool LessAndEqualCompareIntInt(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareIntLong(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareIntFloat(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareIntDouble(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareLongInt(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareLongLong(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareLongFloat(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareLongDouble(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareFloatInt(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareFloatLong(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareFloatFloat(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareFloatDouble(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareDoubleInt(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareDoubleLong(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareDoubleFloat(ExpressionEvalParameters &);
extern __device__ bool LessAndEqualCompareDoubleDouble(ExpressionEvalParameters &);

extern __device__ bool ContainsOperator(ExpressionEvalParameters &);

extern __device__ bool InvalidOperator(ExpressionEvalParameters &);

extern __device__ int AddExpressionInt(ExpressionEvalParameters &);
extern __device__ int MinExpressionInt(ExpressionEvalParameters &);
extern __device__ int MulExpressionInt(ExpressionEvalParameters &);
extern __device__ int DivExpressionInt(ExpressionEvalParameters &);
extern __device__ int ModExpressionInt(ExpressionEvalParameters &);

extern __device__ int64_t AddExpressionLong(ExpressionEvalParameters &);
extern __device__ int64_t MinExpressionLong(ExpressionEvalParameters &);
extern __device__ int64_t MulExpressionLong(ExpressionEvalParameters &);
extern __device__ int64_t DivExpressionLong(ExpressionEvalParameters &);
extern __device__ int64_t ModExpressionLong(ExpressionEvalParameters &);

extern __device__ float AddExpressionFloat(ExpressionEvalParameters &);
extern __device__ float MinExpressionFloat(ExpressionEvalParameters &);
extern __device__ float MulExpressionFloat(ExpressionEvalParameters &);
extern __device__ float DivExpressionFloat(ExpressionEvalParameters &);
extern __device__ float ModExpressionFloat(ExpressionEvalParameters &);

extern __device__ double AddExpressionDouble(ExpressionEvalParameters &);
extern __device__ double MinExpressionDouble(ExpressionEvalParameters &);
extern __device__ double MulExpressionDouble(ExpressionEvalParameters &);
extern __device__ double DivExpressionDouble(ExpressionEvalParameters &);
extern __device__ double ModExpressionDouble(ExpressionEvalParameters &);

extern __device__ bool ExecuteBoolExpression(ExpressionEvalParameters &);
extern __device__ int ExecuteIntExpression(ExpressionEvalParameters &);
extern __device__ int64_t ExecuteLongExpression(ExpressionEvalParameters &);
extern __device__ float ExecuteFloatExpression(ExpressionEvalParameters &);
extern __device__ double ExecuteDoubleExpression(ExpressionEvalParameters &);
extern __device__ const char * ExecuteStringExpression(ExpressionEvalParameters &);

extern __device__ bool Evaluate(ExpressionEvalParameters &);

}


#endif /* GPUJOINKERNELCORE_H_ */
