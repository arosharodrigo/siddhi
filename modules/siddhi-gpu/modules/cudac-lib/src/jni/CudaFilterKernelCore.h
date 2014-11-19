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
#include "CudaEvent.h"

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
typedef bool (*ExecutorFuncPointer)(CudaEvent &, Filter &, int &);

extern __device__ bool cuda_strcmp(const char *s1, const char *s2);

extern __device__ bool NoopOperator(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ bool AndCondition(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool OrCondition(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotCondition(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool BooleanCondition(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ bool EqualCompareBoolBool(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareIntDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareLongDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareFloatDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareDoubleDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool EqualCompareStringString(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
//extern __device__ bool EqualCompareExecutorExecutor(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ bool NotEqualCompareBoolBool(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareIntDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareLongDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareFloatDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareDoubleDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool NotEqualCompareStringString(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ bool GreaterThanCompareIntInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareIntLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareIntFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareIntDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareLongDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareFloatDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterThanCompareDoubleDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ bool LessThanCompareIntInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareIntLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareIntFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareIntDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareLongDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareFloatDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessThanCompareDoubleDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ bool GreaterAndEqualCompareIntInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareIntLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareIntFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareIntDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareLongDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareFloatDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool GreaterAndEqualCompareDoubleDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ bool LessAndEqualCompareIntInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareIntLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareIntFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareIntDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareLongDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareFloatDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ bool LessAndEqualCompareDoubleDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ bool ContainsOperator(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ bool InvalidOperator(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ int AddExpressionInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ int MinExpressionInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ int MulExpressionInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ int DivExpressionInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ int ModExpressionInt(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ long AddExpressionLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ long MinExpressionLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ long MulExpressionLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ long DivExpressionLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ long ModExpressionLong(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ float AddExpressionFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ float MinExpressionFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ float MulExpressionFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ float DivExpressionFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ float ModExpressionFloat(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ double AddExpressionDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ double MinExpressionDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ double MulExpressionDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ double DivExpressionDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ double ModExpressionDouble(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ bool ExecuteBoolExpression(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ int ExecuteIntExpression(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ long ExecuteLongExpression(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ float ExecuteFloatExpression(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ double ExecuteDoubleExpression(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);
extern __device__ const char * ExecuteStringExpression(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

extern __device__ bool Evaluate(CudaEvent & _mEvent, Filter & _mFilter, int & _iCurrentNodeIndex);

};


#endif /* CUDAFILTERKERNELCORE_H_ */
