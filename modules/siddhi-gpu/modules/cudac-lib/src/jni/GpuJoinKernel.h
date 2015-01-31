/*
 * GpuJoinKernel.h
 *
 *  Created on: Jan 31, 2015
 *      Author: prabodha
 */

#ifndef GPUJOINKERNEL_H_
#define GPUJOINKERNEL_H_

#include "GpuKernel.h"

namespace SiddhiGpu
{

class GpuMetaEvent;
class GpuProcessor;
class GpuProcessorContext;
class GpuStreamEventBuffer;
class GpuIntBuffer;

class GpuJoinKernel : public GpuKernel
{
public:
	GpuJoinKernel(GpuProcessor * _pProc, GpuProcessorContext * _pContext, int _iThreadBlockSize,
			int _iLeftWindowSize, int _iRightWindowSize, FILE * _fPLog);
	~GpuJoinKernel();

	bool Initialize(GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize);
	void Process(int _iNumEvents, bool _bLast);
	char * GetResultEventBuffer();
	int GetResultEventBufferSize();

private:
	GpuProcessorContext * p_Context;

	GpuIntBuffer * p_InputEventBuffer;
	GpuStreamEventBuffer * p_ResultEventBuffer;
	GpuStreamEventBuffer * p_OriginalEventBuffer;
	GpuStreamEventBuffer * p_WindowEventBuffer;

	bool b_DeviceSet;
	int i_LeftStraemWindowSize;
	int i_RightStraemWindowSize;
	int i_LeftRemainingCount;
	int i_RightRemainingCount;
};

}


#endif /* GPUJOINKERNEL_H_ */
