/*
 * CudaEvent.cpp
 *
 *  Created on: Oct 23, 2014
 *      Author: prabodha
 */

#include "CudaEvent.h"
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

namespace SiddhiGpu
{

CudaEvent::CudaEvent() :
		ui_Timestamp(0),
		ui_NumAttributes(0)
{
	//printf("Event Default Ctor called\n");
}

CudaEvent::CudaEvent(uint64_t _uiTimeStamp) :
		ui_Timestamp(_uiTimeStamp),
		ui_NumAttributes(0)
{
	//printf("Event Ctor called\n");
}

void CudaEvent::Destroy()
{
	for(unsigned int i=0; i<ui_NumAttributes; ++i)
	{
		if(a_Attributes[i].e_Type == DataType::StringExt)
		{
			delete [] a_Attributes[i].m_Value.z_ExtString;
			a_Attributes[i].m_Value.z_ExtString = NULL;
		}
	}
}

void CudaEvent::AddBoolAttribute(unsigned int _iPos, bool _bValue)
{
	if(_iPos < MAX_ATTR_NUM)
	{
		a_Attributes[_iPos].e_Type = DataType::Boolean;
		a_Attributes[_iPos].m_Value.b_BoolVal = _bValue;
		ui_NumAttributes++;
	}
}

void CudaEvent::AddIntAttribute(unsigned int _iPos, int _iValue)
{
	if(_iPos < MAX_ATTR_NUM)
	{
		a_Attributes[_iPos].e_Type = DataType::Int;
		a_Attributes[_iPos].m_Value.i_IntVal = _iValue;
		ui_NumAttributes++;
	}
}

void CudaEvent::AddLongAttribute(unsigned int _iPos, int64_t _lValue)
{
	if(_iPos < MAX_ATTR_NUM)
	{
		a_Attributes[_iPos].e_Type = DataType::Long;
		a_Attributes[_iPos].m_Value.l_LongVal = _lValue;
		ui_NumAttributes++;
	}
}

void CudaEvent::AddFloatAttribute(unsigned int _iPos, float _fValue)
{
	if(_iPos < MAX_ATTR_NUM)
	{
		a_Attributes[_iPos].e_Type = DataType::Float;
		a_Attributes[_iPos].m_Value.f_FloatVal = _fValue;
		ui_NumAttributes++;
	}
}

void CudaEvent::AddDoubleAttribute(unsigned int _iPos, double _dValue)
{
	if(_iPos < MAX_ATTR_NUM)
	{
		a_Attributes[_iPos].e_Type = DataType::Double;
		a_Attributes[_iPos].m_Value.d_DoubleVal = _dValue;
		ui_NumAttributes++;
	}
}

void CudaEvent::AddStringAttribute(unsigned int _iPos, std::string _sValue)
{
	if(_iPos < MAX_ATTR_NUM)
	{
		if(_sValue.length() > sizeof(int64_t))
		{
			a_Attributes[_iPos].e_Type = DataType::StringExt;
			int iLen = _sValue.length();
			a_Attributes[_iPos].m_Value.z_ExtString = new char[iLen + 1];
			memcpy(a_Attributes[_iPos].m_Value.z_ExtString, _sValue.c_str(), iLen);
			a_Attributes[_iPos].m_Value.z_ExtString[iLen] = 0;
		}
		else
		{
			a_Attributes[_iPos].e_Type = DataType::StringIn;
			memcpy(a_Attributes[_iPos].m_Value.z_StringVal, _sValue.c_str(), sizeof(int64_t));
			a_Attributes[_iPos].m_Value.z_StringVal[sizeof(int64_t)] = 0;
		}
		ui_NumAttributes++;
	}
}

bool CudaEvent::GetBoolAttribute(unsigned int _iPos)
{
	if(_iPos < ui_NumAttributes && a_Attributes[_iPos].e_Type == DataType::Boolean)
	{
		return a_Attributes[_iPos].m_Value.b_BoolVal;
	}
	return false;
}

int CudaEvent::GetIntAttribute(unsigned int _iPos)
{
	if(_iPos < ui_NumAttributes && a_Attributes[_iPos].e_Type == DataType::Int)
	{
		return a_Attributes[_iPos].m_Value.i_IntVal;
	}
	return INT_MIN;
}

int64_t CudaEvent::GetLongAttribute(unsigned int _iPos)
{
	if(_iPos < ui_NumAttributes && a_Attributes[_iPos].e_Type == DataType::Long)
	{
		return a_Attributes[_iPos].m_Value.l_LongVal;
	}
	return LLONG_MIN;
}

float CudaEvent::GetFloatAttribute(unsigned int _iPos)
{
	if(_iPos < ui_NumAttributes && a_Attributes[_iPos].e_Type == DataType::Float)
	{
		return a_Attributes[_iPos].m_Value.f_FloatVal;
	}
	return FLT_MIN;
}

double CudaEvent::GetDoubleAttribute(unsigned int _iPos)
{
	if(_iPos < ui_NumAttributes && a_Attributes[_iPos].e_Type == DataType::Double)
	{
		return a_Attributes[_iPos].m_Value.d_DoubleVal;
	}
	return DBL_MIN;
}

const char * CudaEvent::GetStringAttribute(unsigned int _iPos)
{
	if(_iPos < ui_NumAttributes)
	{
		if(a_Attributes[_iPos].e_Type == DataType::StringIn)
			return a_Attributes[_iPos].m_Value.z_StringVal;

		if(a_Attributes[_iPos].e_Type == DataType::StringExt)
			return a_Attributes[_iPos].m_Value.z_ExtString;
	}
	return NULL;
}

void CudaEvent::Print(FILE * _fp)
{
	fprintf(_fp, "OnEvent : Time=%" PRIu64 " NumAttribs=%d {", ui_Timestamp, ui_NumAttributes);
	for(unsigned int i=0; i<ui_NumAttributes; ++i)
	{
		fprintf(_fp, "(%d Type=%d ", i, a_Attributes[i].e_Type);
		switch(a_Attributes[i].e_Type)
		{
		case DataType::Int:
		{
			fprintf(_fp, "Value=%d) ", a_Attributes[i].m_Value.i_IntVal);
		}
		break;
		case DataType::Long:
		{
			fprintf(_fp, "Value=%ld) ", a_Attributes[i].m_Value.l_LongVal);
		}
		break;
		case DataType::Boolean:
		{
			fprintf(_fp, "Value=%d) ", a_Attributes[i].m_Value.b_BoolVal);
		}
		break;
		case DataType::Float:
		{
			fprintf(_fp, "Value=%f) ", a_Attributes[i].m_Value.f_FloatVal);
		}
		break;
		case DataType::Double:
		{
			fprintf(_fp, "Value=%f) ", a_Attributes[i].m_Value.d_DoubleVal);
		}
		break;
		case DataType::StringIn:
		{
			fprintf(_fp, "Value=%s) ", a_Attributes[i].m_Value.z_StringVal);
		}
		break;
		case DataType::StringExt:
		{
			fprintf(_fp, "Value=%s) ", a_Attributes[i].m_Value.z_ExtString);
		}
		break;
		default:
			break;
		}
	}
	fprintf(_fp, "}\n");
}

};
