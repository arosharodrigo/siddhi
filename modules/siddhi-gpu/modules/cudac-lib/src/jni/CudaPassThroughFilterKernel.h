/*
 * CudaPassThroughFilterKernel.h
 *
 *  Created on: Nov 9, 2014
 *      Author: prabodha
 */

#ifndef CUDAPASSTHROUGHFILTERKERNEL_H_
#define CUDAPASSTHROUGHFILTERKERNEL_H_

#include "CudaFilterKernel.h"

namespace SiddhiGpu
{

class CudaPassThroughFilterKernel : public CudaFilterKernel
{
public:
	CudaPassThroughFilterKernel(GpuEventConsumer * _pConsumer, FILE * _fpLog);
	virtual ~CudaPassThroughFilterKernel();

	void Initialize();

	void ProcessEvents();

	void AddEvent(const CudaEvent * _pEvent);
	void AddAndProcessEvents(CudaEvent ** _apEvent, int _iEventCount);

	void AddFilterToDevice(Filter * _pFilter);
	void CopyFiltersToDevice();
};

};


#endif /* CUDAPASSTHROUGHFILTERKERNEL_H_ */
