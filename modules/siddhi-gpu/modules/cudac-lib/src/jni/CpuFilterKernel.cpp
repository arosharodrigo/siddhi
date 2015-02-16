/*
 * CpuFilterKernel.cpp
 *
 *  Created on: Dec 28, 2014
 *      Author: prabodha
 */

#include "CpuFilterKernel.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "GpuFilterProcessor.h"
#include "GpuMetaEvent.h"

namespace SiddhiCpu
{

bool cuda_strcmp(const char *s1, const char *s2)
{
	for ( ; *s1==*s2; ++s1, ++s2) {
		if (*s1=='\0') return true;
	}
	return false;
}

bool cuda_prefix(char *s1, char *s2)
{
	for ( ; *s1==*s2; ++s1, ++s2) {
		if (*(s2+1)=='\0') return true;
	}
	return false;
}

bool cuda_contains(const char *s1, const char *s2)
{
	int size1 = 0;
	int size2 = 0;

	while (s1[size1]!='\0')
		size1++;

	while (s2[size2]!='\0')
		size2++;

	if (size1==size2)
		return cuda_strcmp(s1, s2);

	if (size1<size2)
		return false;

	for (int i=0; i<size1-size2+1; i++)
	{
		bool failed = false;
		for (int j=0; j<size2; j++)
		{
			if (s1[i+j-1]!=s2[j])
			{
				failed = true;
				break;
			}
		}
		if (! failed)
			return true;
	}
	return false;
}

// ========================= INT ==============================================

int AddExpressionInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ AddExpressionInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) +
			ExecuteIntExpression(_rParameters));
}

int MinExpressionInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ MinExpressionInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) -
			ExecuteIntExpression(_rParameters));
}

int MulExpressionInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ MulExpressionInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) *
			ExecuteIntExpression(_rParameters));
}

int DivExpressionInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ DivExpressionInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) /
			ExecuteIntExpression(_rParameters));
}

int ModExpressionInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ ModExpressionInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteIntExpression(_rParameters) %
			ExecuteIntExpression(_rParameters));
}

// ========================= LONG ==============================================

int64_t AddExpressionLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ AddExpressionLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) +
			ExecuteLongExpression(_rParameters));
}

int64_t MinExpressionLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ MinExpressionLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) -
			ExecuteLongExpression(_rParameters));
}

int64_t MulExpressionLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ MulExpressionLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) *
			ExecuteLongExpression(_rParameters));
}

int64_t DivExpressionLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ DivExpressionLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) /
			ExecuteLongExpression(_rParameters));
}

int64_t ModExpressionLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ ModExpressionLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteLongExpression(_rParameters) %
			ExecuteLongExpression(_rParameters));
}


// ========================= FLOAT ==============================================

float AddExpressionFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ AddExpressionFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteFloatExpression(_rParameters) +
			ExecuteFloatExpression(_rParameters));
}

float MinExpressionFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ MinExpressionFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteFloatExpression(_rParameters) -
			ExecuteFloatExpression(_rParameters));
}

float MulExpressionFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ MulExpressionFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteFloatExpression(_rParameters) *
			ExecuteFloatExpression(_rParameters));
}

float DivExpressionFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ DivExpressionFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteFloatExpression(_rParameters) /
			ExecuteFloatExpression(_rParameters));
}

float ModExpressionFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ ModExpressionFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return fmod(ExecuteFloatExpression(_rParameters),
			ExecuteFloatExpression(_rParameters));
}

// ========================= DOUBLE ===========================================

double AddExpressionDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ AddExpressionDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteDoubleExpression(_rParameters) +
			ExecuteDoubleExpression(_rParameters));
}

double MinExpressionDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ MinExpressionDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteDoubleExpression(_rParameters) -
			ExecuteDoubleExpression(_rParameters));
}

double MulExpressionDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ MulExpressionDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteDoubleExpression(_rParameters) *
			ExecuteDoubleExpression(_rParameters));
}

double DivExpressionDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ DivExpressionDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return (ExecuteDoubleExpression(_rParameters) /
			ExecuteDoubleExpression(_rParameters));
}

double ModExpressionDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ ModExpressionDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return fmod(ExecuteDoubleExpression(_rParameters),
				ExecuteDoubleExpression(_rParameters));
}

/// ============================ CONDITION EXECUTORS ==========================

bool InvalidOperator(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ InvalidOperator [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
	return false;
}

bool NoopOperator(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NoopOperator [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
	return true;
}

// Equal operators

bool EqualCompareBoolBool(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareBoolBool [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

bool EqualCompareIntInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareIntInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

bool EqualCompareIntLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareIntLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) == ExecuteLongExpression(_rParameters));
}

bool EqualCompareIntFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareIntFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) == ExecuteFloatExpression(_rParameters));
}

bool EqualCompareIntDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareIntDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) == ExecuteDoubleExpression(_rParameters));
}

bool EqualCompareLongInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareLongInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

bool EqualCompareLongLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareLongLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) == ExecuteLongExpression(_rParameters));
}

bool EqualCompareLongFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareLongFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) == ExecuteFloatExpression(_rParameters));
}

bool EqualCompareLongDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareLongDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) == ExecuteDoubleExpression(_rParameters));
}


bool EqualCompareFloatInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareFloatInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

bool EqualCompareFloatLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareFloatLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) == ExecuteLongExpression(_rParameters));
}

bool EqualCompareFloatFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareFloatFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) == ExecuteFloatExpression(_rParameters));
}

bool EqualCompareFloatDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareFloatDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) == ExecuteDoubleExpression(_rParameters));
}

bool EqualCompareDoubleInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareDoubleInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) == ExecuteIntExpression(_rParameters));
}

bool EqualCompareDoubleLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareDoubleLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) == ExecuteLongExpression(_rParameters));
}

bool EqualCompareDoubleFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareDoubleFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) == ExecuteFloatExpression(_rParameters));
}

bool EqualCompareDoubleDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareDoubleDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) == ExecuteDoubleExpression(_rParameters));
}

bool EqualCompareStringString(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ EqualCompareStringString [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (cuda_strcmp(ExecuteStringExpression(_rParameters), ExecuteStringExpression(_rParameters)));
}

/// ============================================================================
// NotEqual operator

bool NotEqualCompareBoolBool(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareBoolBool [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

bool NotEqualCompareIntInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareIntInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

bool NotEqualCompareIntLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareIntLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) != ExecuteLongExpression(_rParameters));
}

bool NotEqualCompareIntFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareIntFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) != ExecuteFloatExpression(_rParameters));
}

bool NotEqualCompareIntDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareIntDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) != ExecuteDoubleExpression(_rParameters));
}

bool NotEqualCompareLongInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareLongInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

bool NotEqualCompareLongLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareLongLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) != ExecuteLongExpression(_rParameters));
}

bool NotEqualCompareLongFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareLongFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) != ExecuteFloatExpression(_rParameters));
}

bool NotEqualCompareLongDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareLongDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) != ExecuteDoubleExpression(_rParameters));
}

bool NotEqualCompareFloatInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareFloatInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

bool NotEqualCompareFloatLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareFloatLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) != ExecuteLongExpression(_rParameters));
}

bool NotEqualCompareFloatFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareFloatFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) != ExecuteFloatExpression(_rParameters));
}

bool NotEqualCompareFloatDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareFloatDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) != ExecuteDoubleExpression(_rParameters));
}

bool NotEqualCompareDoubleInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareDoubleInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) != ExecuteIntExpression(_rParameters));
}

bool NotEqualCompareDoubleLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareDoubleLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) != ExecuteLongExpression(_rParameters));
}

bool NotEqualCompareDoubleFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareDoubleFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) != ExecuteFloatExpression(_rParameters));
}

bool NotEqualCompareDoubleDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareDoubleDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) != ExecuteDoubleExpression(_rParameters));
}

bool NotEqualCompareStringString(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotEqualCompareStringString [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (!cuda_strcmp(ExecuteStringExpression(_rParameters),ExecuteStringExpression(_rParameters)));
}

/// ============================================================================

// GreaterThan operator


bool GreaterThanCompareIntInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareIntInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) > ExecuteIntExpression(_rParameters));
}

bool GreaterThanCompareIntLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareIntLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) > ExecuteLongExpression(_rParameters));
}

bool GreaterThanCompareIntFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareIntFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) > ExecuteFloatExpression(_rParameters));
}

bool GreaterThanCompareIntDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareIntDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) > ExecuteDoubleExpression(_rParameters));
}

bool GreaterThanCompareLongInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareLongInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) > ExecuteIntExpression(_rParameters));
}

bool GreaterThanCompareLongLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareLongLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) > ExecuteLongExpression(_rParameters));
}

bool GreaterThanCompareLongFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareLongFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) > ExecuteFloatExpression(_rParameters));
}

bool GreaterThanCompareLongDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareLongDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) > ExecuteDoubleExpression(_rParameters));
}

bool GreaterThanCompareFloatInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareFloatInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) > ExecuteIntExpression(_rParameters));
}

bool GreaterThanCompareFloatLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareFloatLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) > ExecuteLongExpression(_rParameters));
}

bool GreaterThanCompareFloatFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareFloatFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) > ExecuteFloatExpression(_rParameters));
}

bool GreaterThanCompareFloatDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareFloatDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) > ExecuteDoubleExpression(_rParameters));
}

bool GreaterThanCompareDoubleInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareDoubleInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) > ExecuteIntExpression(_rParameters));
}

bool GreaterThanCompareDoubleLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareDoubleLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) > ExecuteLongExpression(_rParameters));
}

bool GreaterThanCompareDoubleFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareDoubleFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) > ExecuteFloatExpression(_rParameters));
}

bool GreaterThanCompareDoubleDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterThanCompareDoubleDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) > ExecuteDoubleExpression(_rParameters));
}

/// ============================================================================
// LessThan operator


bool LessThanCompareIntInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareIntInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) < ExecuteIntExpression(_rParameters));
}

bool LessThanCompareIntLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareIntLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) < ExecuteLongExpression(_rParameters));
}

bool LessThanCompareIntFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareIntFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) < ExecuteFloatExpression(_rParameters));
}

bool LessThanCompareIntDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareIntDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) < ExecuteDoubleExpression(_rParameters));
}

bool LessThanCompareLongInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareLongInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) < ExecuteIntExpression(_rParameters));
}

bool LessThanCompareLongLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareLongLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) < ExecuteLongExpression(_rParameters));
}

bool LessThanCompareLongFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareLongFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) < ExecuteFloatExpression(_rParameters));
}

bool LessThanCompareLongDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareLongDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) < ExecuteDoubleExpression(_rParameters));
}

bool LessThanCompareFloatInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareFloatInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) < ExecuteIntExpression(_rParameters));
}

bool LessThanCompareFloatLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareFloatLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) < ExecuteLongExpression(_rParameters));
}

bool LessThanCompareFloatFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareFloatFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) < ExecuteFloatExpression(_rParameters));
}

bool LessThanCompareFloatDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareFloatDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) < ExecuteDoubleExpression(_rParameters));
}

bool LessThanCompareDoubleInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareDoubleInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) < ExecuteIntExpression(_rParameters));
}

bool LessThanCompareDoubleLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareDoubleLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) < ExecuteLongExpression(_rParameters));
}

bool LessThanCompareDoubleFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareDoubleFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) < ExecuteFloatExpression(_rParameters));
}

bool LessThanCompareDoubleDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessThanCompareDoubleDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) < ExecuteDoubleExpression(_rParameters));
}

/// ============================================================================
// GreaterAndEqual operator

bool GreaterAndEqualCompareIntInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareIntInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) >= ExecuteIntExpression(_rParameters));
}

bool GreaterAndEqualCompareIntLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareIntLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) >= ExecuteLongExpression(_rParameters));
}

bool GreaterAndEqualCompareIntFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareIntFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) >= ExecuteFloatExpression(_rParameters));
}

bool GreaterAndEqualCompareIntDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareIntDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) >= ExecuteDoubleExpression(_rParameters));
}

bool GreaterAndEqualCompareLongInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareLongInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) >= ExecuteIntExpression(_rParameters));
}

bool GreaterAndEqualCompareLongLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareLongLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) >= ExecuteLongExpression(_rParameters));
}

bool GreaterAndEqualCompareLongFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareLongFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) >= ExecuteFloatExpression(_rParameters));
}

bool GreaterAndEqualCompareLongDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareLongDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) >= ExecuteDoubleExpression(_rParameters));
}

bool GreaterAndEqualCompareFloatInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareFloatInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) >= ExecuteIntExpression(_rParameters));
}

bool GreaterAndEqualCompareFloatLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareFloatLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) >= ExecuteLongExpression(_rParameters));
}

bool GreaterAndEqualCompareFloatFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareFloatFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) >= ExecuteFloatExpression(_rParameters));
}

bool GreaterAndEqualCompareFloatDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareFloatDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) >= ExecuteDoubleExpression(_rParameters));
}

bool GreaterAndEqualCompareDoubleInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareDoubleInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) >= ExecuteIntExpression(_rParameters));
}

bool GreaterAndEqualCompareDoubleLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareDoubleLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) >= ExecuteLongExpression(_rParameters));
}

bool GreaterAndEqualCompareDoubleFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareDoubleFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) >= ExecuteFloatExpression(_rParameters));
}

bool GreaterAndEqualCompareDoubleDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ GreaterAndEqualCompareDoubleDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) >= ExecuteDoubleExpression(_rParameters));
}

/// ============================================================================
// LessAndEqual operator

bool LessAndEqualCompareIntInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareIntInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) <= ExecuteIntExpression(_rParameters));
}

bool LessAndEqualCompareIntLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareIntLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) <= ExecuteLongExpression(_rParameters));
}

bool LessAndEqualCompareIntFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareIntFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) <= ExecuteFloatExpression(_rParameters));
}

bool LessAndEqualCompareIntDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareIntDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteIntExpression(_rParameters) <= ExecuteDoubleExpression(_rParameters));
}

bool LessAndEqualCompareLongInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareLongInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) <= ExecuteIntExpression(_rParameters));
}

bool LessAndEqualCompareLongLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareLongLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) <= ExecuteLongExpression(_rParameters));
}

bool LessAndEqualCompareLongFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareLongFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) <= ExecuteFloatExpression(_rParameters));
}

bool LessAndEqualCompareLongDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareLongDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteLongExpression(_rParameters) <= ExecuteDoubleExpression(_rParameters));
}

bool LessAndEqualCompareFloatInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareFloatInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) <= ExecuteIntExpression(_rParameters));
}

bool LessAndEqualCompareFloatLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareFloatLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) <= ExecuteLongExpression(_rParameters));
}

bool LessAndEqualCompareFloatFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareFloatFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) <= ExecuteFloatExpression(_rParameters));
}

bool LessAndEqualCompareFloatDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareFloatDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteFloatExpression(_rParameters) <= ExecuteDoubleExpression(_rParameters));
}

bool LessAndEqualCompareDoubleInt(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareDoubleInt [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) <= ExecuteIntExpression(_rParameters));
}

bool LessAndEqualCompareDoubleLong(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareDoubleLong [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) <= ExecuteLongExpression(_rParameters));
}

bool LessAndEqualCompareDoubleFloat(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareDoubleFloat [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) <= ExecuteFloatExpression(_rParameters));
}

bool LessAndEqualCompareDoubleDouble(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ LessAndEqualCompareDoubleDouble [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (ExecuteDoubleExpression(_rParameters) <= ExecuteDoubleExpression(_rParameters));
}

bool ContainsOperator(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ ContainsOperator [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (cuda_contains(ExecuteStringExpression(_rParameters), ExecuteStringExpression(_rParameters)));
}


/// ============================================================================

bool ExecuteBoolExpression(FilterEvalParameters & _rParameters)
{
	SiddhiGpu::ExecutorNode & mExecutorNode = _rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

	if(mExecutorNode.e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(mExecutorNode.e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(mExecutorNode.m_ConstValue.e_Type == SiddhiGpu::DataType::Boolean)
			{
				fprintf(_rParameters.fp_Log, "{ ExecuteBoolExpression [Event=%p|Index=%d|ConstValue=%d] } ",
					_rParameters.p_Event, _rParameters.i_CurrentIndex,
					mExecutorNode.m_ConstValue.m_Value.b_BoolVal);

				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.b_BoolVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			// if filter data type matches event attribute data type, return attribute value
			if(mExecutorNode.m_VarValue.e_Type == SiddhiGpu::DataType::Boolean &&
					_rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Boolean)
			{
				// get attribute value
				int16_t i;
				memcpy(&i, _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 2);

				fprintf(_rParameters.fp_Log, "{ ExecuteBoolExpression [Event=%p|Index=%d|VarPos=%d|VarValue=%d] } ",
						_rParameters.p_Event, _rParameters.i_CurrentIndex,
						mExecutorNode.m_VarValue.i_AttributePosition, i);

				_rParameters.i_CurrentIndex++;
				return i;
			}
		}
		break;
		default:
			break;
		}
	}

	_rParameters.i_CurrentIndex++;
	return false;
}

int ExecuteIntExpression(FilterEvalParameters & _rParameters)
{
	SiddhiGpu::ExecutorNode & mExecutorNode = _rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

	if(mExecutorNode.e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(mExecutorNode.e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(mExecutorNode.m_ConstValue.e_Type == SiddhiGpu::DataType::Int)
			{
				fprintf(_rParameters.fp_Log, "{ ExecuteIntExpression [Event=%p|Index=%d|ConstValue=%d] } ",
					_rParameters.p_Event, _rParameters.i_CurrentIndex,
					mExecutorNode.m_ConstValue.m_Value.i_IntVal);

				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.i_IntVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(mExecutorNode.m_VarValue.e_Type == SiddhiGpu::DataType::Int &&
					_rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Int)
			{
				int32_t i;
				memcpy(&i, _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 4);

				fprintf(_rParameters.fp_Log, "{ ExecuteIntExpression [Event=%p|Index=%d|VarPos=%d|VarValue=%d] } ",
					_rParameters.p_Event, _rParameters.i_CurrentIndex,
					mExecutorNode.m_VarValue.i_AttributePosition, i);

				_rParameters.i_CurrentIndex++;
				return i;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_ADD_INT:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteIntExpression [Event=%p|Index=%d|AddExpressionInt] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return AddExpressionInt(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_SUB_INT:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteIntExpression [Event=%p|Index=%d|MinExpressionInt] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return MinExpressionInt(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_MUL_INT:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteIntExpression [Event=%p|Index=%d|MulExpressionInt] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return MulExpressionInt(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_DIV_INT:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteIntExpression [Event=%p|Index=%d|DivExpressionInt] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return DivExpressionInt(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_MOD_INT:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteIntExpression [Event=%p|Index=%d|ModExpressionInt] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return ModExpressionInt(_rParameters);
		}
		default:
			break;
		}
	}

	fprintf(_rParameters.fp_Log, "{ ExecuteIntExpression [Event=%p|Index=%d|DefaultValue=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex, INT_MIN);

	_rParameters.i_CurrentIndex++;
	return INT_MIN;
}

int64_t ExecuteLongExpression(FilterEvalParameters & _rParameters)
{
	SiddhiGpu::ExecutorNode & mExecutorNode = _rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

	if(mExecutorNode.e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(mExecutorNode.e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(mExecutorNode.m_ConstValue.e_Type == SiddhiGpu::DataType::Long)
			{
				fprintf(_rParameters.fp_Log, "{ ExecuteLongExpression [Event=%p|Index=%d|ConstValue=%" PRIi64 "] } ",
					_rParameters.p_Event, _rParameters.i_CurrentIndex,
					mExecutorNode.m_ConstValue.m_Value.l_LongVal);

				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.l_LongVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(mExecutorNode.m_VarValue.e_Type == SiddhiGpu::DataType::Long &&
					_rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Long)
			{
				int64_t i;
				memcpy(&i, _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 8);

				fprintf(_rParameters.fp_Log, "{ ExecuteLongExpression [Event=%p|Index=%d|VarPos=%d|VarValue=%" PRIi64 "] } ",
					_rParameters.p_Event, _rParameters.i_CurrentIndex,
					mExecutorNode.m_VarValue.i_AttributePosition, i);

				_rParameters.i_CurrentIndex++;
				return i;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_ADD_LONG:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteLongExpression [Event=%p|Index=%d|AddExpressionLong] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return AddExpressionLong(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_SUB_LONG:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteLongExpression [Event=%p|Index=%d|MinExpressionLong] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return MinExpressionLong(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_MUL_LONG:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteLongExpression [Event=%p|Index=%d|MulExpressionLong] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return MulExpressionLong(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_DIV_LONG:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteLongExpression [Event=%p|Index=%d|DivExpressionLong] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return DivExpressionLong(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_MOD_LONG:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteLongExpression [Event=%p|Index=%d|ModExpressionLong] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return ModExpressionLong(_rParameters);
		}
		default:
			break;
		}
	}

	fprintf(_rParameters.fp_Log, "{ ExecuteLongExpression [Event=%p|Index=%d|DefaultValue=%lli] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex, LLONG_MIN);

	_rParameters.i_CurrentIndex++;
	return LLONG_MIN;
}

float ExecuteFloatExpression(FilterEvalParameters & _rParameters)
{
	SiddhiGpu::ExecutorNode & mExecutorNode = _rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

	if(mExecutorNode.e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(mExecutorNode.e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(mExecutorNode.m_ConstValue.e_Type == SiddhiGpu::DataType::Float)
			{
				fprintf(_rParameters.fp_Log, "{ ExecuteFloatExpression [Event=%p|Index=%d|ConstValue=%f] } ",
					_rParameters.p_Event, _rParameters.i_CurrentIndex,
					mExecutorNode.m_ConstValue.m_Value.f_FloatVal);

				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.f_FloatVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(mExecutorNode.m_VarValue.e_Type == SiddhiGpu::DataType::Float &&
					_rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Float)
			{
				float f;
				memcpy(&f, _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 4);

				fprintf(_rParameters.fp_Log, "{ ExecuteFloatExpression [Event=%p|Index=%d|VarPos=%d|VarValue=%f] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex,
						mExecutorNode.m_VarValue.i_AttributePosition, f);

				_rParameters.i_CurrentIndex++;
				return f;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_ADD_FLOAT:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteFloatExpression [Event=%p|Index=%d|AddExpressionFloat] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return AddExpressionFloat(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_SUB_FLOAT:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteFloatExpression [Event=%p|Index=%d|MinExpressionFloat] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return MinExpressionFloat(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_MUL_FLOAT:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteFloatExpression [Event=%p|Index=%d|MulExpressionFloat] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return MulExpressionFloat(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_DIV_FLOAT:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteFloatExpression [Event=%p|Index=%d|DivExpressionFloat] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return DivExpressionFloat(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_MOD_FLOAT:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteFloatExpression [Event=%p|Index=%d|ModExpressionFloat] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return ModExpressionFloat(_rParameters);
		}
		default:
			break;
		}
	}

	fprintf(_rParameters.fp_Log, "{ ExecuteFloatExpression [Event=%p|Index=%d|DefaultValue=%f] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex, FLT_MIN);

	_rParameters.i_CurrentIndex++;
	return FLT_MIN;
}

double ExecuteDoubleExpression(FilterEvalParameters & _rParameters)
{
	SiddhiGpu::ExecutorNode & mExecutorNode = _rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

	if(mExecutorNode.e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(mExecutorNode.e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(mExecutorNode.m_ConstValue.e_Type == SiddhiGpu::DataType::Double)
			{

				fprintf(_rParameters.fp_Log, "{ ExecuteDoubleExpression [Event=%p|Index=%d|ConstValue=%f] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex,
					mExecutorNode.m_ConstValue.m_Value.d_DoubleVal);

				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.d_DoubleVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(mExecutorNode.m_VarValue.e_Type == SiddhiGpu::DataType::Double &&
					_rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Double)
			{
				double f;
				memcpy(&f, _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 8);

				fprintf(_rParameters.fp_Log, "{ ExecuteDoubleExpression [Event=%p|Index=%d|VarPos=%d|VarValue=%f] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex,
					mExecutorNode.m_VarValue.i_AttributePosition, f);

				_rParameters.i_CurrentIndex++;
				return f;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_ADD_DOUBLE:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteDoubleExpression [Event=%p|Index=%d|AddExpressionDouble] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return AddExpressionDouble(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_SUB_DOUBLE:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteDoubleExpression [Event=%p|Index=%d|MinExpressionDouble] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return MinExpressionDouble(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_MUL_DOUBLE:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteDoubleExpression [Event=%p|Index=%d|MulExpressionDouble] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return MulExpressionDouble(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_DIV_DOUBLE:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteDoubleExpression [Event=%p|Index=%d|DivExpressionDouble] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return DivExpressionDouble(_rParameters);
		}
		case SiddhiGpu::EXPRESSION_MOD_DOUBLE:
		{
			fprintf(_rParameters.fp_Log, "{ ExecuteDoubleExpression [Event=%p|Index=%d|ModExpressionDouble] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);
			return ModExpressionDouble(_rParameters);
		}
		default:
			break;
		}
	}

	fprintf(_rParameters.fp_Log, "{ ExecuteDoubleExpression [Event=%p|Index=%d|DefaultValue=%f] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex, DBL_MIN);

	_rParameters.i_CurrentIndex++;
	return DBL_MIN;
}

const char * ExecuteStringExpression(FilterEvalParameters & _rParameters)
{
	SiddhiGpu::ExecutorNode & mExecutorNode = _rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex];

	if(mExecutorNode.e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(mExecutorNode.e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(mExecutorNode.m_ConstValue.e_Type == SiddhiGpu::DataType::StringIn)
			{
				fprintf(_rParameters.fp_Log, "{ ExecuteStringExpression [Event=%p|Index=%d|ConstInValue=%s] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex,
					mExecutorNode.m_ConstValue.m_Value.z_StringVal);

				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.z_StringVal;
			}
			else if(mExecutorNode.m_ConstValue.e_Type == SiddhiGpu::DataType::StringExt)
			{
				fprintf(_rParameters.fp_Log, "{ ExecuteStringExpression [Event=%p|Index=%d|ConstExtValue=%s] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex,
					mExecutorNode.m_ConstValue.m_Value.z_ExtString);

				_rParameters.i_CurrentIndex++;
				return mExecutorNode.m_ConstValue.m_Value.z_ExtString;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(mExecutorNode.m_VarValue.e_Type == SiddhiGpu::DataType::StringIn &&
					_rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::StringIn)
			{
				int16_t i;
				memcpy(&i, _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position, 2);
				char * z = _rParameters.p_Event + _rParameters.p_Meta->p_Attributes[mExecutorNode.m_VarValue.i_AttributePosition].i_Position + 2;
				z[i] = 0;

				fprintf(_rParameters.fp_Log, "{ ExecuteStringExpression [Event=%p|Index=%d|VarPos=%d|VarLen=%d|VarValue=%s] } ",
					_rParameters.p_Event, _rParameters.i_CurrentIndex,
					mExecutorNode.m_VarValue.i_AttributePosition, i, z);

				_rParameters.i_CurrentIndex++;
				return z;
			}
		}
		break;
		default:
			break;
		}
	}

	fprintf(_rParameters.fp_Log, "{ ExecuteStringExpression [Event=%p|Index=%d|DefaultValue=NULL] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	_rParameters.i_CurrentIndex++;
	return NULL;
}

// ==================================================================================================

bool AndCondition(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ AndCondition [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (Evaluate(_rParameters) && Evaluate(_rParameters));
}

bool OrCondition(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ OrCondition [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (Evaluate(_rParameters) || Evaluate(_rParameters));
}

bool NotCondition(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ NotCondition [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (!Evaluate(_rParameters));
}

bool BooleanCondition(FilterEvalParameters & _rParameters)
{
	fprintf(_rParameters.fp_Log, "{ BooleanCondition [Event=%p|Index=%d] } ", _rParameters.p_Event, _rParameters.i_CurrentIndex);

	return (Evaluate(_rParameters));
}

// =========================================

// set all the executor functions here
ExecutorFuncPointer mExecutors[SiddhiGpu::EXECUTOR_CONDITION_COUNT] = {
		NoopOperator,

		AndCondition,
		OrCondition,
		NotCondition,
		BooleanCondition,

		EqualCompareBoolBool,
		EqualCompareIntInt,
		EqualCompareIntLong,
		EqualCompareIntFloat,
		EqualCompareIntDouble,
		EqualCompareLongInt,
		EqualCompareLongLong,
		EqualCompareLongFloat,
		EqualCompareLongDouble,
		EqualCompareFloatInt,
		EqualCompareFloatLong,
		EqualCompareFloatFloat,
		EqualCompareFloatDouble,
		EqualCompareDoubleInt,
		EqualCompareDoubleLong,
		EqualCompareDoubleFloat,
		EqualCompareDoubleDouble,
		EqualCompareStringString,

		NotEqualCompareBoolBool,
		NotEqualCompareIntInt,
		NotEqualCompareIntLong,
		NotEqualCompareIntFloat,
		NotEqualCompareIntDouble,
		NotEqualCompareLongInt,
		NotEqualCompareLongLong,
		NotEqualCompareLongFloat,
		NotEqualCompareLongDouble,
		NotEqualCompareFloatInt,
		NotEqualCompareFloatLong,
		NotEqualCompareFloatFloat,
		NotEqualCompareFloatDouble,
		NotEqualCompareDoubleInt,
		NotEqualCompareDoubleLong,
		NotEqualCompareDoubleFloat,
		NotEqualCompareDoubleDouble,
		NotEqualCompareStringString,

		GreaterThanCompareIntInt,
		GreaterThanCompareIntLong,
		GreaterThanCompareIntFloat,
		GreaterThanCompareIntDouble,
		GreaterThanCompareLongInt,
		GreaterThanCompareLongLong,
		GreaterThanCompareLongFloat,
		GreaterThanCompareLongDouble,
		GreaterThanCompareFloatInt,
		GreaterThanCompareFloatLong,
		GreaterThanCompareFloatFloat,
		GreaterThanCompareFloatDouble,
		GreaterThanCompareDoubleInt,
		GreaterThanCompareDoubleLong,
		GreaterThanCompareDoubleFloat,
		GreaterThanCompareDoubleDouble,

		LessThanCompareIntInt,
		LessThanCompareIntLong,
		LessThanCompareIntFloat,
		LessThanCompareIntDouble,
		LessThanCompareLongInt,
		LessThanCompareLongLong,
		LessThanCompareLongFloat,
		LessThanCompareLongDouble,
		LessThanCompareFloatInt,
		LessThanCompareFloatLong,
		LessThanCompareFloatFloat,
		LessThanCompareFloatDouble,
		LessThanCompareDoubleInt,
		LessThanCompareDoubleLong,
		LessThanCompareDoubleFloat,
		LessThanCompareDoubleDouble,

		GreaterAndEqualCompareIntInt,
		GreaterAndEqualCompareIntLong,
		GreaterAndEqualCompareIntFloat,
		GreaterAndEqualCompareIntDouble,
		GreaterAndEqualCompareLongInt,
		GreaterAndEqualCompareLongLong,
		GreaterAndEqualCompareLongFloat,
		GreaterAndEqualCompareLongDouble,
		GreaterAndEqualCompareFloatInt,
		GreaterAndEqualCompareFloatLong,
		GreaterAndEqualCompareFloatFloat,
		GreaterAndEqualCompareFloatDouble,
		GreaterAndEqualCompareDoubleInt,
		GreaterAndEqualCompareDoubleLong,
		GreaterAndEqualCompareDoubleFloat,
		GreaterAndEqualCompareDoubleDouble,

		LessAndEqualCompareIntInt,
		LessAndEqualCompareIntLong,
		LessAndEqualCompareIntFloat,
		LessAndEqualCompareIntDouble,
		LessAndEqualCompareLongInt,
		LessAndEqualCompareLongLong,
		LessAndEqualCompareLongFloat,
		LessAndEqualCompareLongDouble,
		LessAndEqualCompareFloatInt,
		LessAndEqualCompareFloatLong,
		LessAndEqualCompareFloatFloat,
		LessAndEqualCompareFloatDouble,
		LessAndEqualCompareDoubleInt,
		LessAndEqualCompareDoubleLong,
		LessAndEqualCompareDoubleFloat,
		LessAndEqualCompareDoubleDouble,

		ContainsOperator,

		InvalidOperator,
};

// =========================================

// evaluate event with an executor tree
bool Evaluate(FilterEvalParameters & _rParameters)
{
	return (*mExecutors[_rParameters.p_Filter->ap_ExecutorNodes[_rParameters.i_CurrentIndex++].e_ConditionType])(_rParameters);
}

};



