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

namespace SiddhiGpu
{

// executor function pointer type
typedef bool (*ExecutorFuncPointer)(Filter &, EventMeta &, char *, int &);

extern bool cuda_strcmp(const char *s1, const char *s2);

extern bool NoopOperator(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool AndCondition(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool OrCondition(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotCondition(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool BooleanCondition(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool EqualCompareBoolBool(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool EqualCompareStringString(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
//extern bool EqualCompareExecutorExecutor(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool NotEqualCompareBoolBool(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool NotEqualCompareStringString(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool GreaterThanCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterThanCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool LessThanCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessThanCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool GreaterAndEqualCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool GreaterAndEqualCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool LessAndEqualCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern bool LessAndEqualCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool ContainsOperator(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool InvalidOperator(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern int AddExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int MinExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int MulExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int DivExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int ModExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern int64_t AddExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int64_t MinExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int64_t MulExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int64_t DivExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int64_t ModExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern float AddExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern float MinExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern float MulExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern float DivExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern float ModExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern double AddExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern double MinExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern double MulExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern double DivExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern double ModExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool ExecuteBoolExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int ExecuteIntExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern int64_t ExecuteLongExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern float ExecuteFloatExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern double ExecuteDoubleExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern const char * ExecuteStringExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern bool Evaluate(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

};

#endif /* CPUFILTERKERNEL_H_ */
