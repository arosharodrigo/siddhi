/*
 * Value.h
 *
 *  Created on: Jan 20, 2015
 *      Author: prabodha
 */

#ifndef VALUE_H_
#define VALUE_H_

#include <stdint.h>

namespace SiddhiGpu
{

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

}


#endif /* VALUE_H_ */
