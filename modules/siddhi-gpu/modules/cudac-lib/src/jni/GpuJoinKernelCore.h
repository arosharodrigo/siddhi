/*
 * GpuJoinKernelCore.h
 *
 *  Created on: Feb 10, 2015
 *      Author: prabodha
 */

#ifndef GPUJOINKERNELCORE_H_
#define GPUJOINKERNELCORE_H_

#include "GpuKernelDataTypes.h"

namespace SiddhiGpu
{

// executor function pointer type
typedef bool (*OnCompareFuncPointer)(GpuKernelFilter &, GpuKernelMetaEvent *, char *, GpuKernelMetaEvent *, char *, int &);


extern __device__ bool NoopOperator(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ bool AndCondition(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool OrCondition(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotCondition(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool BooleanCondition(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ bool EqualCompareBoolBool(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareStringString(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
//extern __device__ bool EqualCompareExecutorExecutor(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ bool NotEqualCompareBoolBool(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareStringString(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ bool GreaterThanCompareIntInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareIntLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareIntFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareIntDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ bool LessThanCompareIntInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareIntLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareIntFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareIntDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ bool GreaterAndEqualCompareIntInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareIntLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareIntFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareIntDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ bool LessAndEqualCompareIntInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareIntLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareIntFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareIntDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ bool ContainsOperator(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ bool InvalidOperator(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ int AddExpressionInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ int MinExpressionInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ int MulExpressionInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ int DivExpressionInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ int ModExpressionInt(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ int64_t AddExpressionLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t MinExpressionLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t MulExpressionLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t DivExpressionLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t ModExpressionLong(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ float AddExpressionFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ float MinExpressionFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ float MulExpressionFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ float DivExpressionFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ float ModExpressionFloat(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ double AddExpressionDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ double MinExpressionDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ double MulExpressionDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ double DivExpressionDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ double ModExpressionDouble(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ bool ExecuteBoolExpression(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ int ExecuteIntExpression(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t ExecuteLongExpression(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ float ExecuteFloatExpression(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ double ExecuteDoubleExpression(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);
extern __device__ const char * ExecuteStringExpression(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

extern __device__ bool Evaluate(GpuKernelFilter & _mOnCompare, GpuKernelMetaEvent * _pLeftMeta, char * _pLeftEvent, GpuKernelMetaEvent * _pRightMeta, char * _pRightEvent, int & _iCurrentNodeIndex);

}


#endif /* GPUJOINKERNELCORE_H_ */
