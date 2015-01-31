/*
 * CudaKernelBase.cpp
 *
 *  Created on: Nov 10, 2014
 *      Author: prabodha
 */


#include "old/CudaKernelBase.h"

namespace SiddhiGpu
{

CudaKernelBase::CudaKernelBase(GpuEventConsumer * _pConsumer, FILE * _fpLog) :
		p_EventConsumer(_pConsumer),
		fp_Log(_fpLog),
		i_SizeOfEvent(0),
		i_ResultsBufferPosition(0),
		i_EventMetaBufferPosition(0),
		i_EventDataBufferPosition(0)
{

}

CudaKernelBase::~CudaKernelBase()
{

}


};

