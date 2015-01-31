/*
 * GpuCudaTimer.h
 *
 *  Created on: Jan 30, 2015
 *      Author: prabodha
 */

#ifndef CUDAGPUTIMER_H_
#define CUDAGPUTIMER_H_

namespace SiddhiGpu
{

class GpuCudaTimer
{
	cudaEvent_t start;
	cudaEvent_t end;
public:
	GpuCudaTimer();
	~GpuCudaTimer();

	float MillisecondsElapsed();
	float SecondsElapsed();

};
}


#endif /* CUDAGPUTIMER_H_ */
