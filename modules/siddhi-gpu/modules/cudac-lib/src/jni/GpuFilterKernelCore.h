/*
 * GpuFilterKernelCore.h
 *
 *  Created on: Jan 27, 2015
 *      Author: prabodha
 */

#ifndef GPUFILTERKERNELCORE_H_
#define GPUFILTERKERNELCORE_H_

#include "GpuKernelDataTypes.h"
#include "GpuCudaCommon.h"

namespace SiddhiGpu
{

typedef struct FilterEvalParameters
{
//	GpuKernelFilter      * p_Filter;
	GpuKernelMetaEvent   * p_Meta;
	char                 * p_Event;
	int                    i_CurrentIndex;
} FilterEvalParameters;

// executor function pointer type
typedef bool (*ExecutorFuncPointer)(FilterEvalParameters &);

extern __device__ bool NoopOperator(FilterEvalParameters &);

extern __device__ bool AndCondition(FilterEvalParameters &);
extern __device__ bool OrCondition(FilterEvalParameters &);
extern __device__ bool NotCondition(FilterEvalParameters &);
extern __device__ bool BooleanCondition(FilterEvalParameters &);

extern __device__ bool EqualCompareBoolBool(FilterEvalParameters &);
extern __device__ bool EqualCompareIntInt(FilterEvalParameters &);
extern __device__ bool EqualCompareIntLong(FilterEvalParameters &);
extern __device__ bool EqualCompareIntFloat(FilterEvalParameters &);
extern __device__ bool EqualCompareIntDouble(FilterEvalParameters &);
extern __device__ bool EqualCompareLongInt(FilterEvalParameters &);
extern __device__ bool EqualCompareLongLong(FilterEvalParameters &);
extern __device__ bool EqualCompareLongFloat(FilterEvalParameters &);
extern __device__ bool EqualCompareLongDouble(FilterEvalParameters &);
extern __device__ bool EqualCompareFloatInt(FilterEvalParameters &);
extern __device__ bool EqualCompareFloatLong(FilterEvalParameters &);
extern __device__ bool EqualCompareFloatFloat(FilterEvalParameters &);
extern __device__ bool EqualCompareFloatDouble(FilterEvalParameters &);
extern __device__ bool EqualCompareDoubleInt(FilterEvalParameters &);
extern __device__ bool EqualCompareDoubleLong(FilterEvalParameters &);
extern __device__ bool EqualCompareDoubleFloat(FilterEvalParameters &);
extern __device__ bool EqualCompareDoubleDouble(FilterEvalParameters &);
extern __device__ bool EqualCompareStringString(FilterEvalParameters &);

extern __device__ bool NotEqualCompareBoolBool(FilterEvalParameters &);
extern __device__ bool NotEqualCompareIntInt(FilterEvalParameters &);
extern __device__ bool NotEqualCompareIntLong(FilterEvalParameters &);
extern __device__ bool NotEqualCompareIntFloat(FilterEvalParameters &);
extern __device__ bool NotEqualCompareIntDouble(FilterEvalParameters &);
extern __device__ bool NotEqualCompareLongInt(FilterEvalParameters &);
extern __device__ bool NotEqualCompareLongLong(FilterEvalParameters &);
extern __device__ bool NotEqualCompareLongFloat(FilterEvalParameters &);
extern __device__ bool NotEqualCompareLongDouble(FilterEvalParameters &);
extern __device__ bool NotEqualCompareFloatInt(FilterEvalParameters &);
extern __device__ bool NotEqualCompareFloatLong(FilterEvalParameters &);
extern __device__ bool NotEqualCompareFloatFloat(FilterEvalParameters &);
extern __device__ bool NotEqualCompareFloatDouble(FilterEvalParameters &);
extern __device__ bool NotEqualCompareDoubleInt(FilterEvalParameters &);
extern __device__ bool NotEqualCompareDoubleLong(FilterEvalParameters &);
extern __device__ bool NotEqualCompareDoubleFloat(FilterEvalParameters &);
extern __device__ bool NotEqualCompareDoubleDouble(FilterEvalParameters &);
extern __device__ bool NotEqualCompareStringString(FilterEvalParameters &);

extern __device__ bool GreaterThanCompareIntInt(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareIntLong(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareIntFloat(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareIntDouble(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareLongInt(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareLongLong(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareLongFloat(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareLongDouble(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareFloatInt(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareFloatLong(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareFloatFloat(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareFloatDouble(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareDoubleInt(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareDoubleLong(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareDoubleFloat(FilterEvalParameters &);
extern __device__ bool GreaterThanCompareDoubleDouble(FilterEvalParameters &);

extern __device__ bool LessThanCompareIntInt(FilterEvalParameters &);
extern __device__ bool LessThanCompareIntLong(FilterEvalParameters &);
extern __device__ bool LessThanCompareIntFloat(FilterEvalParameters &);
extern __device__ bool LessThanCompareIntDouble(FilterEvalParameters &);
extern __device__ bool LessThanCompareLongInt(FilterEvalParameters &);
extern __device__ bool LessThanCompareLongLong(FilterEvalParameters &);
extern __device__ bool LessThanCompareLongFloat(FilterEvalParameters &);
extern __device__ bool LessThanCompareLongDouble(FilterEvalParameters &);
extern __device__ bool LessThanCompareFloatInt(FilterEvalParameters &);
extern __device__ bool LessThanCompareFloatLong(FilterEvalParameters &);
extern __device__ bool LessThanCompareFloatFloat(FilterEvalParameters &);
extern __device__ bool LessThanCompareFloatDouble(FilterEvalParameters &);
extern __device__ bool LessThanCompareDoubleInt(FilterEvalParameters &);
extern __device__ bool LessThanCompareDoubleLong(FilterEvalParameters &);
extern __device__ bool LessThanCompareDoubleFloat(FilterEvalParameters &);
extern __device__ bool LessThanCompareDoubleDouble(FilterEvalParameters &);

extern __device__ bool GreaterAndEqualCompareIntInt(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareIntLong(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareIntFloat(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareIntDouble(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareLongInt(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareLongLong(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareLongFloat(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareLongDouble(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareFloatInt(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareFloatLong(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareFloatFloat(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareFloatDouble(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareDoubleInt(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareDoubleLong(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareDoubleFloat(FilterEvalParameters &);
extern __device__ bool GreaterAndEqualCompareDoubleDouble(FilterEvalParameters &);

extern __device__ bool LessAndEqualCompareIntInt(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareIntLong(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareIntFloat(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareIntDouble(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareLongInt(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareLongLong(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareLongFloat(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareLongDouble(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareFloatInt(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareFloatLong(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareFloatFloat(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareFloatDouble(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareDoubleInt(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareDoubleLong(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareDoubleFloat(FilterEvalParameters &);
extern __device__ bool LessAndEqualCompareDoubleDouble(FilterEvalParameters &);

extern __device__ bool ContainsOperator(FilterEvalParameters &);

extern __device__ bool InvalidOperator(FilterEvalParameters &);

extern __device__ int AddExpressionInt(FilterEvalParameters &);
extern __device__ int MinExpressionInt(FilterEvalParameters &);
extern __device__ int MulExpressionInt(FilterEvalParameters &);
extern __device__ int DivExpressionInt(FilterEvalParameters &);
extern __device__ int ModExpressionInt(FilterEvalParameters &);

extern __device__ int64_t AddExpressionLong(FilterEvalParameters &);
extern __device__ int64_t MinExpressionLong(FilterEvalParameters &);
extern __device__ int64_t MulExpressionLong(FilterEvalParameters &);
extern __device__ int64_t DivExpressionLong(FilterEvalParameters &);
extern __device__ int64_t ModExpressionLong(FilterEvalParameters &);

extern __device__ float AddExpressionFloat(FilterEvalParameters &);
extern __device__ float MinExpressionFloat(FilterEvalParameters &);
extern __device__ float MulExpressionFloat(FilterEvalParameters &);
extern __device__ float DivExpressionFloat(FilterEvalParameters &);
extern __device__ float ModExpressionFloat(FilterEvalParameters &);

extern __device__ double AddExpressionDouble(FilterEvalParameters &);
extern __device__ double MinExpressionDouble(FilterEvalParameters &);
extern __device__ double MulExpressionDouble(FilterEvalParameters &);
extern __device__ double DivExpressionDouble(FilterEvalParameters &);
extern __device__ double ModExpressionDouble(FilterEvalParameters &);

extern __device__ bool ExecuteBoolExpression(FilterEvalParameters &);
extern __device__ int ExecuteIntExpression(FilterEvalParameters &);
extern __device__ int64_t ExecuteLongExpression(FilterEvalParameters &);
extern __device__ float ExecuteFloatExpression(FilterEvalParameters &);
extern __device__ double ExecuteDoubleExpression(FilterEvalParameters &);
extern __device__ const char * ExecuteStringExpression(FilterEvalParameters &);

extern __device__ bool Evaluate(FilterEvalParameters &);

}


#endif /* GPUFILTERKERNELCORE_H_ */
