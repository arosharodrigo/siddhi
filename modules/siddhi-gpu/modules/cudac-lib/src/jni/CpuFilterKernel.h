/*
 * CpuFilterKernel.h
 *
 *  Created on: Dec 28, 2014
 *      Author: prabodha
 */

#ifndef _CPUFILTERKERNEL_H_
#define _CPUFILTERKERNEL_H_

#include <stdint.h>
#include <stdio.h>

namespace SiddhiGpu { class GpuFilterProcessor; }
namespace SiddhiGpu { class GpuMetaEvent; }

namespace SiddhiCpu
{

typedef struct FilterEvalParameters
{
	SiddhiGpu::GpuFilterProcessor * p_Filter;
	SiddhiGpu::GpuMetaEvent       * p_Meta;
	char                          * p_Event;
	int                             i_CurrentIndex;
	FILE                          * fp_Log;
} FilterEvalParameters;

// executor function pointer type
typedef bool (*ExecutorFuncPointer)(FilterEvalParameters &);

extern bool NoopOperator(FilterEvalParameters &);

extern bool AndCondition(FilterEvalParameters &);
extern bool OrCondition(FilterEvalParameters &);
extern bool NotCondition(FilterEvalParameters &);
extern bool BooleanCondition(FilterEvalParameters &);

extern bool EqualCompareBoolBool(FilterEvalParameters &);
extern bool EqualCompareIntInt(FilterEvalParameters &);
extern bool EqualCompareIntLong(FilterEvalParameters &);
extern bool EqualCompareIntFloat(FilterEvalParameters &);
extern bool EqualCompareIntDouble(FilterEvalParameters &);
extern bool EqualCompareLongInt(FilterEvalParameters &);
extern bool EqualCompareLongLong(FilterEvalParameters &);
extern bool EqualCompareLongFloat(FilterEvalParameters &);
extern bool EqualCompareLongDouble(FilterEvalParameters &);
extern bool EqualCompareFloatInt(FilterEvalParameters &);
extern bool EqualCompareFloatLong(FilterEvalParameters &);
extern bool EqualCompareFloatFloat(FilterEvalParameters &);
extern bool EqualCompareFloatDouble(FilterEvalParameters &);
extern bool EqualCompareDoubleInt(FilterEvalParameters &);
extern bool EqualCompareDoubleLong(FilterEvalParameters &);
extern bool EqualCompareDoubleFloat(FilterEvalParameters &);
extern bool EqualCompareDoubleDouble(FilterEvalParameters &);
extern bool EqualCompareStringString(FilterEvalParameters &);

extern bool NotEqualCompareBoolBool(FilterEvalParameters &);
extern bool NotEqualCompareIntInt(FilterEvalParameters &);
extern bool NotEqualCompareIntLong(FilterEvalParameters &);
extern bool NotEqualCompareIntFloat(FilterEvalParameters &);
extern bool NotEqualCompareIntDouble(FilterEvalParameters &);
extern bool NotEqualCompareLongInt(FilterEvalParameters &);
extern bool NotEqualCompareLongLong(FilterEvalParameters &);
extern bool NotEqualCompareLongFloat(FilterEvalParameters &);
extern bool NotEqualCompareLongDouble(FilterEvalParameters &);
extern bool NotEqualCompareFloatInt(FilterEvalParameters &);
extern bool NotEqualCompareFloatLong(FilterEvalParameters &);
extern bool NotEqualCompareFloatFloat(FilterEvalParameters &);
extern bool NotEqualCompareFloatDouble(FilterEvalParameters &);
extern bool NotEqualCompareDoubleInt(FilterEvalParameters &);
extern bool NotEqualCompareDoubleLong(FilterEvalParameters &);
extern bool NotEqualCompareDoubleFloat(FilterEvalParameters &);
extern bool NotEqualCompareDoubleDouble(FilterEvalParameters &);
extern bool NotEqualCompareStringString(FilterEvalParameters &);

extern bool GreaterThanCompareIntInt(FilterEvalParameters &);
extern bool GreaterThanCompareIntLong(FilterEvalParameters &);
extern bool GreaterThanCompareIntFloat(FilterEvalParameters &);
extern bool GreaterThanCompareIntDouble(FilterEvalParameters &);
extern bool GreaterThanCompareLongInt(FilterEvalParameters &);
extern bool GreaterThanCompareLongLong(FilterEvalParameters &);
extern bool GreaterThanCompareLongFloat(FilterEvalParameters &);
extern bool GreaterThanCompareLongDouble(FilterEvalParameters &);
extern bool GreaterThanCompareFloatInt(FilterEvalParameters &);
extern bool GreaterThanCompareFloatLong(FilterEvalParameters &);
extern bool GreaterThanCompareFloatFloat(FilterEvalParameters &);
extern bool GreaterThanCompareFloatDouble(FilterEvalParameters &);
extern bool GreaterThanCompareDoubleInt(FilterEvalParameters &);
extern bool GreaterThanCompareDoubleLong(FilterEvalParameters &);
extern bool GreaterThanCompareDoubleFloat(FilterEvalParameters &);
extern bool GreaterThanCompareDoubleDouble(FilterEvalParameters &);

extern bool LessThanCompareIntInt(FilterEvalParameters &);
extern bool LessThanCompareIntLong(FilterEvalParameters &);
extern bool LessThanCompareIntFloat(FilterEvalParameters &);
extern bool LessThanCompareIntDouble(FilterEvalParameters &);
extern bool LessThanCompareLongInt(FilterEvalParameters &);
extern bool LessThanCompareLongLong(FilterEvalParameters &);
extern bool LessThanCompareLongFloat(FilterEvalParameters &);
extern bool LessThanCompareLongDouble(FilterEvalParameters &);
extern bool LessThanCompareFloatInt(FilterEvalParameters &);
extern bool LessThanCompareFloatLong(FilterEvalParameters &);
extern bool LessThanCompareFloatFloat(FilterEvalParameters &);
extern bool LessThanCompareFloatDouble(FilterEvalParameters &);
extern bool LessThanCompareDoubleInt(FilterEvalParameters &);
extern bool LessThanCompareDoubleLong(FilterEvalParameters &);
extern bool LessThanCompareDoubleFloat(FilterEvalParameters &);
extern bool LessThanCompareDoubleDouble(FilterEvalParameters &);

extern bool GreaterAndEqualCompareIntInt(FilterEvalParameters &);
extern bool GreaterAndEqualCompareIntLong(FilterEvalParameters &);
extern bool GreaterAndEqualCompareIntFloat(FilterEvalParameters &);
extern bool GreaterAndEqualCompareIntDouble(FilterEvalParameters &);
extern bool GreaterAndEqualCompareLongInt(FilterEvalParameters &);
extern bool GreaterAndEqualCompareLongLong(FilterEvalParameters &);
extern bool GreaterAndEqualCompareLongFloat(FilterEvalParameters &);
extern bool GreaterAndEqualCompareLongDouble(FilterEvalParameters &);
extern bool GreaterAndEqualCompareFloatInt(FilterEvalParameters &);
extern bool GreaterAndEqualCompareFloatLong(FilterEvalParameters &);
extern bool GreaterAndEqualCompareFloatFloat(FilterEvalParameters &);
extern bool GreaterAndEqualCompareFloatDouble(FilterEvalParameters &);
extern bool GreaterAndEqualCompareDoubleInt(FilterEvalParameters &);
extern bool GreaterAndEqualCompareDoubleLong(FilterEvalParameters &);
extern bool GreaterAndEqualCompareDoubleFloat(FilterEvalParameters &);
extern bool GreaterAndEqualCompareDoubleDouble(FilterEvalParameters &);

extern bool LessAndEqualCompareIntInt(FilterEvalParameters &);
extern bool LessAndEqualCompareIntLong(FilterEvalParameters &);
extern bool LessAndEqualCompareIntFloat(FilterEvalParameters &);
extern bool LessAndEqualCompareIntDouble(FilterEvalParameters &);
extern bool LessAndEqualCompareLongInt(FilterEvalParameters &);
extern bool LessAndEqualCompareLongLong(FilterEvalParameters &);
extern bool LessAndEqualCompareLongFloat(FilterEvalParameters &);
extern bool LessAndEqualCompareLongDouble(FilterEvalParameters &);
extern bool LessAndEqualCompareFloatInt(FilterEvalParameters &);
extern bool LessAndEqualCompareFloatLong(FilterEvalParameters &);
extern bool LessAndEqualCompareFloatFloat(FilterEvalParameters &);
extern bool LessAndEqualCompareFloatDouble(FilterEvalParameters &);
extern bool LessAndEqualCompareDoubleInt(FilterEvalParameters &);
extern bool LessAndEqualCompareDoubleLong(FilterEvalParameters &);
extern bool LessAndEqualCompareDoubleFloat(FilterEvalParameters &);
extern bool LessAndEqualCompareDoubleDouble(FilterEvalParameters &);

extern bool ContainsOperator(FilterEvalParameters &);

extern bool InvalidOperator(FilterEvalParameters &);

extern int AddExpressionInt(FilterEvalParameters &);
extern int MinExpressionInt(FilterEvalParameters &);
extern int MulExpressionInt(FilterEvalParameters &);
extern int DivExpressionInt(FilterEvalParameters &);
extern int ModExpressionInt(FilterEvalParameters &);

extern int64_t AddExpressionLong(FilterEvalParameters &);
extern int64_t MinExpressionLong(FilterEvalParameters &);
extern int64_t MulExpressionLong(FilterEvalParameters &);
extern int64_t DivExpressionLong(FilterEvalParameters &);
extern int64_t ModExpressionLong(FilterEvalParameters &);

extern float AddExpressionFloat(FilterEvalParameters &);
extern float MinExpressionFloat(FilterEvalParameters &);
extern float MulExpressionFloat(FilterEvalParameters &);
extern float DivExpressionFloat(FilterEvalParameters &);
extern float ModExpressionFloat(FilterEvalParameters &);

extern double AddExpressionDouble(FilterEvalParameters &);
extern double MinExpressionDouble(FilterEvalParameters &);
extern double MulExpressionDouble(FilterEvalParameters &);
extern double DivExpressionDouble(FilterEvalParameters &);
extern double ModExpressionDouble(FilterEvalParameters &);

extern bool ExecuteBoolExpression(FilterEvalParameters &);
extern int ExecuteIntExpression(FilterEvalParameters &);
extern int64_t ExecuteLongExpression(FilterEvalParameters &);
extern float ExecuteFloatExpression(FilterEvalParameters &);
extern double ExecuteDoubleExpression(FilterEvalParameters &);
extern const char * ExecuteStringExpression(FilterEvalParameters &);

extern bool Evaluate(FilterEvalParameters &);

};

#endif
