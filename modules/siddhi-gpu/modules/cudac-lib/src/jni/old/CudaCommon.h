/*
 * CudaCommon.h
 *
 *  Created on: Dec 26, 2014
 *      Author: prabodha
 */

#ifndef OLD_CUDACOMMON_H_
#define OLD_CUDACOMMON_H_

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

};


#endif /* OLD_CUDACOMMON_H_ */
