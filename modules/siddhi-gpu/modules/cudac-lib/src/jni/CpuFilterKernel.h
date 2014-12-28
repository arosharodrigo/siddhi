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
typedef bool (*ExecutorFuncPointer)(SiddhiGpu::Filter &, EventMeta &, char *, int &, FILE *);

extern bool cuda_strcmp(const char *s1, const char *s2);

extern bool NoopOperator(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern bool AndCondition(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool OrCondition(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotCondition(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool BooleanCondition(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern bool EqualCompareBoolBool(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareIntInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareIntLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareIntFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareIntDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareLongInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareLongLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareLongFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareLongDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareFloatInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareFloatLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareFloatFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareFloatDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareDoubleInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareDoubleLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareDoubleFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareDoubleDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool EqualCompareStringString(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
//extern bool EqualCompareExecutorExecutor(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern bool NotEqualCompareBoolBool(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareIntInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareIntLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareIntFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareIntDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareLongInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareLongLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareLongFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareLongDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareFloatInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareFloatLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareFloatFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareFloatDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareDoubleInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareDoubleLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareDoubleFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareDoubleDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool NotEqualCompareStringString(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern bool GreaterThanCompareIntInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareIntLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareIntFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareIntDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareLongInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareLongLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareLongFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareLongDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareFloatInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareFloatLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareFloatFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareFloatDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareDoubleInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareDoubleLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareDoubleFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterThanCompareDoubleDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern bool LessThanCompareIntInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareIntLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareIntFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareIntDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareLongInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareLongLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareLongFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareLongDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareFloatInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareFloatLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareFloatFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareFloatDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareDoubleInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareDoubleLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareDoubleFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessThanCompareDoubleDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern bool GreaterAndEqualCompareIntInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareIntLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareIntFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareIntDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareLongInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareLongLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareLongFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareLongDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareFloatInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareFloatLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareFloatFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareFloatDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareDoubleInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareDoubleLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareDoubleFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool GreaterAndEqualCompareDoubleDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern bool LessAndEqualCompareIntInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareIntLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareIntFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareIntDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareLongInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareLongLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareLongFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareLongDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareFloatInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareFloatLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareFloatFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareFloatDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareDoubleInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareDoubleLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareDoubleFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern bool LessAndEqualCompareDoubleDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern bool ContainsOperator(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern bool InvalidOperator(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern int AddExpressionInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern int MinExpressionInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern int MulExpressionInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern int DivExpressionInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern int ModExpressionInt(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern int64_t AddExpressionLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern int64_t MinExpressionLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern int64_t MulExpressionLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern int64_t DivExpressionLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern int64_t ModExpressionLong(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern float AddExpressionFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern float MinExpressionFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern float MulExpressionFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern float DivExpressionFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern float ModExpressionFloat(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern double AddExpressionDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern double MinExpressionDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern double MulExpressionDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern double DivExpressionDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern double ModExpressionDouble(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern bool ExecuteBoolExpression(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern int ExecuteIntExpression(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern int64_t ExecuteLongExpression(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern float ExecuteFloatExpression(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern double ExecuteDoubleExpression(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);
extern const char * ExecuteStringExpression(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

extern bool Evaluate(SiddhiGpu::Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp);

};

#endif /* CPUFILTERKERNEL_H_ */
