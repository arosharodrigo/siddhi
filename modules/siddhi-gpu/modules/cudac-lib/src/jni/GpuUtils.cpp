/*
 * GpuUtils.cpp
 *
 *  Created on: Jan 21, 2015
 *      Author: prabodha
 */


#include "GpuUtils.h"
#include <unistd.h>
#include <syscall.h>

namespace SiddhiGpu
{

void GpuUtils::PrintThreadInfo(const char * _zTag, FILE * _fpLog)
{
#ifdef GPU_DEBUG
	pid_t pid = getpid();
	int tid = syscall(__NR_gettid);
	fprintf(_fpLog, "[%s] EventConsumer : PID=%d TID=%d\n", _zTag, pid, tid);
#endif
}

}
