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


};



#endif /* DATATYPES_H_ */
