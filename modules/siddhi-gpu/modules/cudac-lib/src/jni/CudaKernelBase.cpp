/*
 * CudaKernelBase.cpp
 *
 *  Created on: Nov 10, 2014
 *      Author: prabodha
 */


#include "CudaKernelBase.h"

namespace SiddhiGpu
{

CudaKernelBase::CudaKernelBase(GpuEventConsumer * _pConsumer, FILE * _fpLog) :
		p_EventConsumer(_pConsumer),
		fp_Log(_fpLog)
{

}

CudaKernelBase::~CudaKernelBase()
{

}


};

