/*
 * GpuFilterKernelCore.h
 *
 *  Created on: Jan 27, 2015
 *      Author: prabodha
 */

#ifndef GPUFILTERKERNELCORE_H_
#define GPUFILTERKERNELCORE_H_

#include "GpuKernelDataTypes.h"

namespace SiddhiGpu
{

// executor function pointer type
typedef bool (*ExecutorFuncPointer)(GpuKernelFilter &, GpuKernelMetaEvent *, char *, int &);

extern __device__ bool cuda_strcmp(const char *s1, const char *s2);

extern __device__ bool NoopOperator(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool AndCondition(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool OrCondition(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotCondition(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool BooleanCondition(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool EqualCompareBoolBool(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareStringString(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
//extern __device__ bool EqualCompareExecutorExecutor(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool NotEqualCompareBoolBool(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareStringString(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool GreaterThanCompareIntInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareIntLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareIntFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareIntDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool LessThanCompareIntInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareIntLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareIntFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareIntDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool GreaterAndEqualCompareIntInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareIntLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareIntFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareIntDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool LessAndEqualCompareIntInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareIntLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareIntFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareIntDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool ContainsOperator(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool InvalidOperator(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ int AddExpressionInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int MinExpressionInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int MulExpressionInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int DivExpressionInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int ModExpressionInt(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ int64_t AddExpressionLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t MinExpressionLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t MulExpressionLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t DivExpressionLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t ModExpressionLong(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ float AddExpressionFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ float MinExpressionFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ float MulExpressionFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ float DivExpressionFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ float ModExpressionFloat(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ double AddExpressionDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ double MinExpressionDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ double MulExpressionDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ double DivExpressionDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ double ModExpressionDouble(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool ExecuteBoolExpression(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int ExecuteIntExpression(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t ExecuteLongExpression(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ float ExecuteFloatExpression(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ double ExecuteDoubleExpression(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ const char * ExecuteStringExpression(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool Evaluate(GpuKernelFilter & _mFilter, GpuKernelMetaEvent * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

}


#endif /* GPUFILTERKERNELCORE_H_ */
