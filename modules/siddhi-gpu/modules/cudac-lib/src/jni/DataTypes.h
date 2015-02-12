/*
 * DataTypes.h
 *
 *  Created on: Jan 20, 2015
 *      Author: prabodha
 */

#ifndef DATATYPES_H_
#define DATATYPES_H_

#include <stdint.h>

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

#pragma pack(1)

typedef struct AttributeMapping
{
	enum {
		STREAM_INDEX = 0,
		ATTRIBUTE_INDEX = 1
	};

	int from[2];
	int to;
}AttributeMapping;

typedef struct AttributeMappings
{
	AttributeMappings(int _iMappingCount) :
		i_MappingCount(_iMappingCount)
	{
		p_Mappings = new AttributeMapping[_iMappingCount];
	}

	void AddMapping(int _iPos, int _iFromStream, int _iFromAttribute, int _iToAttribute)
	{
		if(_iPos >= 0 && _iPos < i_MappingCount)
		{
			p_Mappings[_iPos].from[AttributeMapping::STREAM_INDEX] = _iFromStream;
			p_Mappings[_iPos].from[AttributeMapping::ATTRIBUTE_INDEX] = _iFromAttribute;
			p_Mappings[_iPos].to = _iToAttribute;
		}
	}

	AttributeMappings * Clone()
	{
		AttributeMappings * pRet = new AttributeMappings(i_MappingCount);
		for(int i=0; i<i_MappingCount; ++i)
		{
			pRet->p_Mappings[i].from[AttributeMapping::STREAM_INDEX] = p_Mappings[i].from[AttributeMapping::STREAM_INDEX];
			pRet->p_Mappings[i].from[AttributeMapping::ATTRIBUTE_INDEX] = p_Mappings[i].from[AttributeMapping::ATTRIBUTE_INDEX];
			pRet->p_Mappings[i].to = p_Mappings[i].to;
		}
		return pRet;
	}

	int i_MappingCount;
	AttributeMapping * p_Mappings;
}AttributeMappings;

#pragma pack()

};



#endif /* DATATYPES_H_ */
