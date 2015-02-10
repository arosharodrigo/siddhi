/*
 * GpuCudaCommon.h
 *
 *  Created on: Feb 11, 2015
 *      Author: prabodha
 */

#ifndef GPUCUDACOMMON_H_
#define GPUCUDACOMMON_H_


namespace SiddhiGpu
{

extern __device__ bool cuda_strcmp(const char *s1, const char *s2);
extern __device__ bool cuda_prefix(char *s1, char *s2);
extern __device__ bool cuda_contains(const char *s1, const char *s2);

}

#endif /* GPUCUDACOMMON_H_ */
