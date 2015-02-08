/*
 * GpuJoinKernel.h
 *
 *  Created on: Jan 31, 2015
 *      Author: prabodha
 */

#ifndef GPUJOINKERNEL_H_
#define GPUJOINKERNEL_H_

#include <stdio.h>
#include "GpuKernel.h"

namespace SiddhiGpu
{

class GpuMetaEvent;
class GpuProcessor;
class GpuProcessorContext;
class GpuStreamEventBuffer;
class GpuIntBuffer;
class GpuRawByteBuffer;
class GpuJoinProcessor;

class GpuJoinKernel : public GpuKernel
{
public:
	GpuJoinKernel(GpuProcessor * _pProc, GpuProcessorContext * _pLeftContext, GpuProcessorContext * _pRightContext,
			int _iThreadBlockSize, int _iLeftWindowSize, int _iRightWindowSize, FILE * _fpLeftLog, FILE * _fpRightLog);
	~GpuJoinKernel();

	bool Initialize(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize);
	void Process(int _iStreamIndex, int &_iNumEvents, bool _bLast);
	char * GetResultEventBuffer();
	int GetResultEventBufferSize();

	int GetLeftInputEventBufferIndex() { return i_LeftInputBufferIndex; }
	void SetLeftInputEventBufferIndex(int _iIndex) { i_LeftInputBufferIndex = _iIndex; }
	int GetRightInputEventBufferIndex() { return i_RightInputBufferIndex; }
	void SetRightInputEventBufferIndex(int _iIndex) { i_RightInputBufferIndex = _iIndex; }

private:
	GpuJoinProcessor * p_JoinProcessor;

	GpuProcessorContext * p_LeftContext;
	GpuProcessorContext * p_RightContext;

	int i_LeftInputBufferIndex;
	int i_RightInputBufferIndex;

	GpuStreamEventBuffer * p_LeftInputEventBuffer;
	GpuStreamEventBuffer * p_RightInputEventBuffer;
	GpuStreamEventBuffer * p_LeftWindowEventBuffer;
	GpuStreamEventBuffer * p_RightWindowEventBuffer;
	GpuRawByteBuffer * p_ResultEventBuffer;

	int i_LeftStreamWindowSize;
	int i_RightStreamWindowSize;
	int i_LeftRemainingCount;
	int i_RightRemainingCount;

	bool b_DeviceSet;
	int i_InitializedStreamCount;
	FILE * fp_LeftLog;
	FILE * fp_RightLog;
};

}


#endif /* GPUJOINKERNEL_H_ */
