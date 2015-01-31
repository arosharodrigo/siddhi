/*
 * CpuFilterKernel.cpp
 *
 *  Created on: Dec 28, 2014
 *      Author: prabodha
 */

#include "old/CpuFilterKernel.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

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

int AddExpressionInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ AddExpressionInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) +
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

int MinExpressionInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ MinExpressionInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) -
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

int MulExpressionInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ MulExpressionInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) *
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

int DivExpressionInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ DivExpressionInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) /
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

int ModExpressionInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ ModExpressionInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) %
			ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

// ========================= LONG ==============================================

int64_t AddExpressionLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ AddExpressionLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) +
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

int64_t MinExpressionLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ MinExpressionLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) -
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

int64_t MulExpressionLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ MulExpressionLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) *
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

int64_t DivExpressionLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ DivExpressionLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) /
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

int64_t ModExpressionLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ ModExpressionLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) %
			ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}


// ========================= FLOAT ==============================================

float AddExpressionFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ AddExpressionFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) +
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

float MinExpressionFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ MinExpressionFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) -
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

float MulExpressionFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ MulExpressionFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) *
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

float DivExpressionFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ DivExpressionFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) /
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

float ModExpressionFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ ModExpressionFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return fmod(ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp),
			ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

// ========================= DOUBLE ===========================================

double AddExpressionDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ AddExpressionDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) +
			ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

double MinExpressionDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ MinExpressionDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) -
			ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

double MulExpressionDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ MulExpressionDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) *
			ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

double DivExpressionDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ DivExpressionDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) /
			ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

double ModExpressionDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ ModExpressionDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return fmod(ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp),
				ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

/// ============================ CONDITION EXECUTORS ==========================

bool InvalidOperator(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ InvalidOperator [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);
	return false;
}

bool NoopOperator(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NoopOperator [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);
	return true;
}

// Equal operators

bool EqualCompareBoolBool(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareBoolBool [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareIntInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareIntInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareIntLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareIntLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareIntFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareIntFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareIntDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareIntDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareLongInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareLongInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareLongLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareLongLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareLongFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareLongFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareLongDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareLongDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}


bool EqualCompareFloatInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareFloatInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareFloatLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareFloatLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareFloatFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareFloatFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareFloatDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareFloatDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareDoubleInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareDoubleInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareDoubleLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareDoubleLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareDoubleFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareDoubleFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareDoubleDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareDoubleDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) == ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool EqualCompareStringString(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ EqualCompareStringString [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (cuda_strcmp(ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp), ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp)));
}

/// ============================================================================
// NotEqual operator

bool NotEqualCompareBoolBool(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareBoolBool [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareIntInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareIntInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareIntLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareIntLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareIntFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareIntFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareIntDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareIntDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareLongInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareLongInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareLongLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareLongLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareLongFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareLongFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareLongDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareLongDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareFloatInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareFloatInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareFloatLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareFloatLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareFloatFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareFloatFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareFloatDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareFloatDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareDoubleInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareDoubleInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareDoubleLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareDoubleLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareDoubleFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareDoubleFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareDoubleDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareDoubleDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) != ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotEqualCompareStringString(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotEqualCompareStringString [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (!cuda_strcmp(ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp),ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp)));
}

/// ============================================================================

// GreaterThan operator


bool GreaterThanCompareIntInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareIntInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareIntLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareIntLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareIntFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareIntFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareIntDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareIntDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareLongInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareLongInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareLongLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareLongLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareLongFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareLongFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareLongDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareLongDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareFloatInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareFloatInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareFloatLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareFloatLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareFloatFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareFloatFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareFloatDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareFloatDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareDoubleInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareDoubleInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareDoubleLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareDoubleLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareDoubleFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareDoubleFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterThanCompareDoubleDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterThanCompareDoubleDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) > ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

/// ============================================================================
// LessThan operator


bool LessThanCompareIntInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareIntInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareIntLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareIntLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareIntFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareIntFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareIntDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareIntDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareLongInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareLongInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareLongLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareLongLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareLongFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareLongFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareLongDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareLongDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareFloatInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareFloatInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareFloatLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareFloatLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareFloatFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareFloatFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareFloatDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareFloatDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareDoubleInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareDoubleInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareDoubleLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareDoubleLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareDoubleFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareDoubleFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessThanCompareDoubleDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessThanCompareDoubleDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) < ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

/// ============================================================================
// GreaterAndEqual operator

bool GreaterAndEqualCompareIntInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareIntInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareIntLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareIntLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareIntFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareIntFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareIntDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareIntDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareLongInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareLongInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareLongLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareLongLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareLongFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareLongFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareLongDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareLongDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareFloatInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareFloatInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareFloatLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareFloatLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareFloatFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareFloatFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareFloatDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareFloatDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareDoubleInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareDoubleInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareDoubleLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareDoubleLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareDoubleFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareDoubleFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool GreaterAndEqualCompareDoubleDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ GreaterAndEqualCompareDoubleDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) >= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

/// ============================================================================
// LessAndEqual operator

bool LessAndEqualCompareIntInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareIntInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareIntLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareIntLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareIntFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareIntFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareIntDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareIntDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareLongInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareLongInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareLongLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareLongLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareLongFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareLongFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareLongDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareLongDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareFloatInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareFloatInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareFloatLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareFloatLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareFloatFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareFloatFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareFloatDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareFloatDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareDoubleInt(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareDoubleInt [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteIntExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareDoubleLong(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareDoubleLong [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteLongExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareDoubleFloat(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareDoubleFloat [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteFloatExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool LessAndEqualCompareDoubleDouble(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ LessAndEqualCompareDoubleDouble [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) <= ExecuteDoubleExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool ContainsOperator(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ ContainsOperator [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (cuda_contains(ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp), ExecuteStringExpression(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp)));
}


/// ============================================================================

bool ExecuteBoolExpression(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::Boolean)
			{
				fprintf(_fp, "{ ExecuteBoolExpression [Event=%p|Index=%d|ConstValue=%d] } ", _pEvent, _iCurrentNodeIndex,
						_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.m_Value.b_BoolVal);

				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.b_BoolVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			// if filter data type matches event attribute data type, return attribute value
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == SiddhiGpu::DataType::Boolean &&
					_pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Boolean)
			{
				// get attribute value
				int16_t i;
				memcpy(&i, _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 2);

				fprintf(_fp, "{ ExecuteBoolExpression [Event=%p|Index=%d|VarPos=%d|VarValue=%d] } ", _pEvent, _iCurrentNodeIndex,
						_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition, i);

				_iCurrentNodeIndex++;
				return i;
			}
		}
		break;
		default:
			break;
		}
	}

	fprintf(_fp, "{ ExecuteBoolExpression [Event=%p|Index=%d|DefaultValue=false] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return false;
}

int ExecuteIntExpression(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::Int)
			{
				fprintf(_fp, "{ ExecuteIntExpression [Event=%p|Index=%d|ConstValue=%d] } ", _pEvent, _iCurrentNodeIndex,
						_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.m_Value.i_IntVal);

				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.i_IntVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == SiddhiGpu::DataType::Int &&
					_pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Int)
			{
				int32_t i;
				memcpy(&i, _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);

				fprintf(_fp, "{ ExecuteIntExpression [Event=%p|Index=%d|VarPos=%d|VarValue=%d] } ", _pEvent, _iCurrentNodeIndex,
						_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition, i);

				_iCurrentNodeIndex++;
				return i;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_ADD_INT:
		{
			fprintf(_fp, "{ ExecuteIntExpression [Event=%p|Index=%d|AddExpressionInt] } ", _pEvent, _iCurrentNodeIndex);
			return AddExpressionInt(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_SUB_INT:
		{
			fprintf(_fp, "{ ExecuteIntExpression [Event=%p|Index=%d|MinExpressionInt] } ", _pEvent, _iCurrentNodeIndex);
			return MinExpressionInt(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_MUL_INT:
		{
			fprintf(_fp, "{ ExecuteIntExpression [Event=%p|Index=%d|MulExpressionInt] } ", _pEvent, _iCurrentNodeIndex);
			return MulExpressionInt(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_DIV_INT:
		{
			fprintf(_fp, "{ ExecuteIntExpression [Event=%p|Index=%d|DivExpressionInt] } ", _pEvent, _iCurrentNodeIndex);
			return DivExpressionInt(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_MOD_INT:
		{
			fprintf(_fp, "{ ExecuteIntExpression [Event=%p|Index=%d|ModExpressionInt] } ", _pEvent, _iCurrentNodeIndex);
			return ModExpressionInt(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		default:
			break;
		}
	}

	fprintf(_fp, "{ ExecuteIntExpression [Event=%p|Index=%d|DefaultValue=%d] } ", _pEvent, _iCurrentNodeIndex, INT_MIN);

	_iCurrentNodeIndex++;
	return INT_MIN;
}

int64_t ExecuteLongExpression(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::Long)
			{
				fprintf(_fp, "{ ExecuteLongExpression [Event=%p|Index=%d|ConstValue=%" PRIi64 "] } ", _pEvent, _iCurrentNodeIndex,
						_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.m_Value.l_LongVal);

				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.l_LongVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == SiddhiGpu::DataType::Long &&
					_pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Long)
			{
				int64_t i;
				memcpy(&i, _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);

				fprintf(_fp, "{ ExecuteLongExpression [Event=%p|Index=%d|VarPos=%d|VarValue=%" PRIi64 "] } ", _pEvent, _iCurrentNodeIndex,
						_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition, i);

				_iCurrentNodeIndex++;
				return i;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_ADD_LONG:
		{
			fprintf(_fp, "{ ExecuteLongExpression [Event=%p|Index=%d|AddExpressionLong] } ", _pEvent, _iCurrentNodeIndex);
			return AddExpressionLong(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_SUB_LONG:
		{
			fprintf(_fp, "{ ExecuteLongExpression [Event=%p|Index=%d|MinExpressionLong] } ", _pEvent, _iCurrentNodeIndex);
			return MinExpressionLong(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_MUL_LONG:
		{
			fprintf(_fp, "{ ExecuteLongExpression [Event=%p|Index=%d|MulExpressionLong] } ", _pEvent, _iCurrentNodeIndex);
			return MulExpressionLong(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_DIV_LONG:
		{
			fprintf(_fp, "{ ExecuteLongExpression [Event=%p|Index=%d|DivExpressionLong] } ", _pEvent, _iCurrentNodeIndex);
			return DivExpressionLong(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_MOD_LONG:
		{
			fprintf(_fp, "{ ExecuteLongExpression [Event=%p|Index=%d|ModExpressionLong] } ", _pEvent, _iCurrentNodeIndex);
			return ModExpressionLong(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		default:
			break;
		}
	}

	fprintf(_fp, "{ ExecuteLongExpression [Event=%p|Index=%d|DefaultValue=%lli] } ", _pEvent, _iCurrentNodeIndex, LLONG_MIN);

	_iCurrentNodeIndex++;
	return LLONG_MIN;
}

float ExecuteFloatExpression(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::Float)
			{
				fprintf(_fp, "{ ExecuteFloatExpression [Event=%p|Index=%d|ConstValue=%f] } ", _pEvent, _iCurrentNodeIndex,
						_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.m_Value.f_FloatVal);

				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.f_FloatVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == SiddhiGpu::DataType::Float &&
					_pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Float)
			{
				float f;
				memcpy(&f, _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 4);

				fprintf(_fp, "{ ExecuteFloatExpression [Event=%p|Index=%d|VarPos=%d|VarValue=%f] } ", _pEvent, _iCurrentNodeIndex,
						_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition, f);

				_iCurrentNodeIndex++;
				return f;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_ADD_FLOAT:
		{
			fprintf(_fp, "{ ExecuteFloatExpression [Event=%p|Index=%d|AddExpressionFloat] } ", _pEvent, _iCurrentNodeIndex);
			return AddExpressionFloat(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_SUB_FLOAT:
		{
			fprintf(_fp, "{ ExecuteFloatExpression [Event=%p|Index=%d|MinExpressionFloat] } ", _pEvent, _iCurrentNodeIndex);
			return MinExpressionFloat(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_MUL_FLOAT:
		{
			fprintf(_fp, "{ ExecuteFloatExpression [Event=%p|Index=%d|MulExpressionFloat] } ", _pEvent, _iCurrentNodeIndex);
			return MulExpressionFloat(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_DIV_FLOAT:
		{
			fprintf(_fp, "{ ExecuteFloatExpression [Event=%p|Index=%d|DivExpressionFloat] } ", _pEvent, _iCurrentNodeIndex);
			return DivExpressionFloat(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_MOD_FLOAT:
		{
			fprintf(_fp, "{ ExecuteFloatExpression [Event=%p|Index=%d|ModExpressionFloat] } ", _pEvent, _iCurrentNodeIndex);
			return ModExpressionFloat(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		default:
			break;
		}
	}

	fprintf(_fp, "{ ExecuteFloatExpression [Event=%p|Index=%d|DefaultValue=%f] } ", _pEvent, _iCurrentNodeIndex, FLT_MIN);

	_iCurrentNodeIndex++;
	return FLT_MIN;
}

double ExecuteDoubleExpression(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::Double)
			{
				fprintf(_fp, "{ ExecuteDoubleExpression [Event=%p|Index=%d|ConstValue=%f] } ", _pEvent, _iCurrentNodeIndex,
						_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.m_Value.d_DoubleVal);

				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.d_DoubleVal;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == SiddhiGpu::DataType::Double &&
					_pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::Double)
			{
				double f;
				memcpy(&f, _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 8);

				fprintf(_fp, "{ ExecuteDoubleExpression [Event=%p|Index=%d|VarPos=%d|VarValue=%f] } ", _pEvent, _iCurrentNodeIndex,
						_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition, f);

				_iCurrentNodeIndex++;
				return f;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_ADD_DOUBLE:
		{
			fprintf(_fp, "{ ExecuteDoubleExpression [Event=%p|Index=%d|AddExpressionDouble] } ", _pEvent, _iCurrentNodeIndex);
			return AddExpressionDouble(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_SUB_DOUBLE:
		{
			fprintf(_fp, "{ ExecuteDoubleExpression [Event=%p|Index=%d|MinExpressionDouble] } ", _pEvent, _iCurrentNodeIndex);
			return MinExpressionDouble(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_MUL_DOUBLE:
		{
			fprintf(_fp, "{ ExecuteDoubleExpression [Event=%p|Index=%d|MulExpressionDouble] } ", _pEvent, _iCurrentNodeIndex);
			return MulExpressionDouble(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_DIV_DOUBLE:
		{
			fprintf(_fp, "{ ExecuteDoubleExpression [Event=%p|Index=%d|DivExpressionDouble] } ", _pEvent, _iCurrentNodeIndex);
			return DivExpressionDouble(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		case SiddhiGpu::EXPRESSION_MOD_DOUBLE:
		{
			fprintf(_fp, "{ ExecuteDoubleExpression [Event=%p|Index=%d|ModExpressionDouble] } ", _pEvent, _iCurrentNodeIndex);
			return ModExpressionDouble(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
		}
		default:
			break;
		}
	}

	fprintf(_fp, "{ ExecuteDoubleExpression [Event=%p|Index=%d|DefaultValue=%f] } ", _pEvent, _iCurrentNodeIndex, DBL_MIN);

	_iCurrentNodeIndex++;
	return DBL_MIN;
}

const char * ExecuteStringExpression(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_NodeType == SiddhiGpu::EXECUTOR_NODE_EXPRESSION)
	{
		switch(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].e_ExpressionType)
		{
		case SiddhiGpu::EXPRESSION_CONST:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::StringIn)
			{
				fprintf(_fp, "{ ExecuteStringExpression [Event=%p|Index=%d|ConstInValue=%s] } ", _pEvent, _iCurrentNodeIndex,
						_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.m_Value.z_StringVal);

				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.z_StringVal;
			}
			else if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.e_Type == SiddhiGpu::DataType::StringExt)
			{
				fprintf(_fp, "{ ExecuteStringExpression [Event=%p|Index=%d|ConstExtValue=%s] } ", _pEvent, _iCurrentNodeIndex,
						_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_ConstValue.m_Value.z_ExtString);

				return _mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].m_ConstValue.m_Value.z_ExtString;
			}
		}
		break;
		case SiddhiGpu::EXPRESSION_VARIABLE:
		{
			if(_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.e_Type == SiddhiGpu::DataType::StringIn &&
					_pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Type == SiddhiGpu::DataType::StringIn)
			{
				int16_t i;
				memcpy(&i, _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position, 2);
				char * z = _pEvent + _pEventMeta->a_Attributes[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition].i_Position + 2;
				z[i] = 0;

				fprintf(_fp, "{ ExecuteStringExpression [Event=%p|Index=%d|VarPos=%d|VarLen=%d|VarValue=%s] } ", _pEvent, _iCurrentNodeIndex,
						_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex].m_VarValue.i_AttributePosition, i, z);

				_iCurrentNodeIndex++;
				return z;
			}
		}
		break;
		default:
			break;
		}
	}

	fprintf(_fp, "{ ExecuteStringExpression [Event=%p|Index=%d|DefaultValue=NULL] } ", _pEvent, _iCurrentNodeIndex);

	_iCurrentNodeIndex++;
	return NULL;
}

// ==================================================================================================

bool AndCondition(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ AndCondition [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) && Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool OrCondition(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ OrCondition [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp) || Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool NotCondition(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ NotCondition [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (!Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
}

bool BooleanCondition(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	fprintf(_fp, "{ BooleanCondition [Event=%p|Index=%d] } ", _pEvent, _iCurrentNodeIndex);

	return (Evaluate(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp));
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
bool Evaluate(SiddhiGpu::Filter& _mFilter, EventMeta * _pEventMeta, char * _pEvent, int & _iCurrentNodeIndex, FILE * _fp)
{
	return (*mExecutors[_mFilter.ap_ExecutorNodes[_iCurrentNodeIndex++].e_ConditionType])(_mFilter, _pEventMeta, _pEvent, _iCurrentNodeIndex, _fp);
}

};



