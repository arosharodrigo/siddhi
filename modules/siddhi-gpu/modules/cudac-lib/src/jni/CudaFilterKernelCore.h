/*
 * CudaFilterKernelCore.h
 *
 *  Created on: Nov 9, 2014
 *      Author: prabodha
 */

#ifndef CUDAFILTERKERNELCORE_H_
#define CUDAFILTERKERNELCORE_H_

#include <stdlib.h>
#include "Filter.h"
#include "CudaCommon.h"
#include "ByteBufferStructs.h"

#include <cuda_profiler_api.h>

namespace SiddhiGpu
{

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error [%s] at line [%d] in file [%s]\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


// executor function pointer type
typedef bool (*ExecutorFuncPointer)(Filter &, EventMeta &, char *, int &);

extern __device__ bool cuda_strcmp(const char *s1, const char *s2);

extern __device__ bool NoopOperator(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool AndCondition(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool OrCondition(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotCondition(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool BooleanCondition(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool EqualCompareBoolBool(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareStringString(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
//extern __device__ bool EqualCompareExecutorExecutor(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool NotEqualCompareBoolBool(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareStringString(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool GreaterThanCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool LessThanCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool GreaterAndEqualCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool LessAndEqualCompareIntInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareIntLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareIntFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareIntDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool ContainsOperator(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool InvalidOperator(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ int AddExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int MinExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int MulExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int DivExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int ModExpressionInt(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ int64_t AddExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t MinExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t MulExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t DivExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t ModExpressionLong(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ float AddExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ float MinExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ float MulExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ float DivExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ float ModExpressionFloat(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ double AddExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ double MinExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ double MulExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ double DivExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ double ModExpressionDouble(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool ExecuteBoolExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int ExecuteIntExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ int64_t ExecuteLongExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ float ExecuteFloatExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ double ExecuteDoubleExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);
extern __device__ const char * ExecuteStringExpression(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

extern __device__ bool Evaluate(Filter & _mFilter, EventMeta & _mEventMeta, char * _pEvent, int & _iCurrentNodeIndex);

};


#endif /* CUDAFILTERKERNELCORE_H_ */
