/*
 * CpuFilterKernel.h
 *
 *  Created on: Dec 28, 2014
 *      Author: prabodha
 */

#ifndef CPUFILTERKERNEL_H_
#define CPUFILTERKERNEL_H_

#include <stdlib.h>
#include "Filter.h"
#include "CudaCommon.h"
#include "ByteBufferStructs.h"

namespace SiddhiCpu
{

// executor function pointer type
typedef bool (*ExecutorFuncPointer)(SiddhiGpu::Filter &, EventMeta &, char *, int &);

extern bool cuda_strcmp(const char *s1, const char *s2);

extern bool NoopOperator(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool AndCondition(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool OrCondition(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotCondition(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool BooleanCondition(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool EqualCompareBoolBool(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareIntInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareIntLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareIntFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareIntDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareLongInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareLongLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareLongFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareLongDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareFloatInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareFloatLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareFloatFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareFloatDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareDoubleInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareDoubleLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareDoubleFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareDoubleDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareStringString(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
//extern bool EqualCompareExecutorExecutor(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool NotEqualCompareBoolBool(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareIntInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareIntLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareIntFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareIntDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareLongInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareLongLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareLongFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareLongDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareFloatInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareFloatLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareFloatFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareFloatDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareDoubleInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareDoubleLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareDoubleFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareDoubleDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareStringString(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool GreaterThanCompareIntInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareIntLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareIntFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareIntDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareLongInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareLongLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareLongFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareLongDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareFloatInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareFloatLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareFloatFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareFloatDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareDoubleInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareDoubleLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareDoubleFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareDoubleDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool LessThanCompareIntInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareIntLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareIntFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareIntDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareLongInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareLongLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareLongFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareLongDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareFloatInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareFloatLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareFloatFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareFloatDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareDoubleInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareDoubleLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareDoubleFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareDoubleDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool GreaterAndEqualCompareIntInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareIntLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareIntFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareIntDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareLongInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareLongLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareLongFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareLongDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareFloatInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareFloatLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareFloatFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareFloatDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareDoubleInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareDoubleLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareDoubleFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareDoubleDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool LessAndEqualCompareIntInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareIntLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareIntFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareIntDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareLongInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareLongLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareLongFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareLongDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareFloatInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareFloatLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareFloatFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareFloatDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareDoubleInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareDoubleLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareDoubleFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareDoubleDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool ContainsOperator(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool InvalidOperator(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern int AddExpressionInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int MinExpressionInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int MulExpressionInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int DivExpressionInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int ModExpressionInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern int64_t AddExpressionLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int64_t MinExpressionLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int64_t MulExpressionLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int64_t DivExpressionLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int64_t ModExpressionLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern float AddExpressionFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern float MinExpressionFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern float MulExpressionFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern float DivExpressionFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern float ModExpressionFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern double AddExpressionDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern double MinExpressionDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern double MulExpressionDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern double DivExpressionDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern double ModExpressionDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool ExecuteBoolExpression(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int ExecuteIntExpression(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int64_t ExecuteLongExpression(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern float ExecuteFloatExpression(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern double ExecuteDoubleExpression(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern const char * ExecuteStringExpression(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool Evaluate(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

};

#endif /* CPUFILTERKERNEL_H_ */
