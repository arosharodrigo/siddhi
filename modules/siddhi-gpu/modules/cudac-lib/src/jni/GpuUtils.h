/*
 * GpuUtils.h
 *
 *  Created on: Jan 21, 2015
 *      Author: prabodha
 */

#ifndef GPUUTILS_H_
#define GPUUTILS_H_

#include <stdio.h>

namespace SiddhiGpu
{

class GpuMetaEvent;

class GpuUtils
{
public:

	static void PrintThreadInfo(const char * _zTag, FILE * _fpLog);
	static void PrintByteBuffer(char * _pEventBuffer, int _iNumEvents, GpuMetaEvent * _pEventMeta, const char * _zTag, FILE * _fpLog);
};

}


#endif /* GPUUTILS_H_ */
