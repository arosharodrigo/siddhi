/*
 * CudaEvent.h
 *
 *  Created on: Oct 23, 2014
 *      Author: prabodha
 */

#ifndef CUDAEVENT_H_
#define CUDAEVENT_H_

#include <string.h>
#include <string>
#include <stdint.h>
#include <stdio.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

namespace SiddhiGpu
{

struct DataType
{
	enum Value
	{
		Int       = 0,
		Long      = 1,
		Boolean   = 2,
		Float     = 3,
		Double    = 4,
		StringIn  = 5,
		StringExt = 6,
		None      = 7
	};

	static const char * GetTypeName(Value _eType)
	{
		switch(_eType)
		{
			case Int: //     = 0,
				return "INT";
			case Long: //    = 1,
				return "LONG";
			case Boolean:
				return "BOOL";
			case Float: //   = 2,
				return "FLOAT";
			case Double://  = 3,
				return "DOUBLE";
			case StringIn: //  = 4,
			case StringExt:
				return "STRING";
			case None: //    = 5
				return "NONE";
		}
		return "NONE";
	}
};

union Values
{
	bool     b_BoolVal;
	int      i_IntVal;
	int64_t  l_LongVal;
	float    f_FloatVal;
	double   d_DoubleVal;
	char    z_StringVal[sizeof(int64_t)]; // set this if strlen < 8
	char *  z_ExtString; // set this if strlen > 8
};

class AttibuteValue
{
public:
	Values m_Value;
	DataType::Value e_Type;
};

class CudaEvent
{
public:
	enum { MAX_ATTR_NUM = 10 };

	CudaEvent();
	CudaEvent(uint64_t _uiTimeStamp);
	void Destroy();

	inline uint64_t GetTimestamp() { return ui_Timestamp; }
	inline unsigned int GetNumAttributes() { return ui_NumAttributes; }

	void AddBoolAttribute(unsigned int _iPos, bool _bValue);
	void AddIntAttribute(unsigned int _iPos, int _iValue);
	void AddLongAttribute(unsigned int _iPos, int64_t _lValue);
	void AddFloatAttribute(unsigned int _iPos, float _fValue);
	void AddDoubleAttribute(unsigned int _iPos, double _dValue);
	void AddStringAttribute(unsigned int _iPos, std::string _sValue);

	bool GetBoolAttribute(unsigned int _iPos);
	int GetIntAttribute(unsigned int _iPos);
	int64_t GetLongAttribute(unsigned int _iPos);
	float GetFloatAttribute(unsigned int _iPos);
	double GetDoubleAttribute(unsigned int _iPos);
	const char * GetStringAttribute(unsigned int _iPos);

	void Print() { Print(stdout); }
	void Print(FILE * _fp);

	uint64_t ui_Timestamp;
	unsigned int ui_NumAttributes;
	AttibuteValue a_Attributes[MAX_ATTR_NUM]; // index based values, use a enum schema to access
};

};

#endif /* CUDAEVENT_H_ */
