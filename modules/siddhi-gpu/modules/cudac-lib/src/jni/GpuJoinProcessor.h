/*
 * GpuJoinProcessor.h
 *
 *  Created on: Jan 31, 2015
 *      Author: prabodha
 */

#ifndef GPUJOINPROCESSOR_H_
#define GPUJOINPROCESSOR_H_

#include <stdio.h>
#include "GpuProcessor.h"

namespace SiddhiGpu
{

class GpuMetaEvent;
class GpuJoinKernel;
class GpuProcessorContext;

class GpuJoinProcessor : public GpuProcessor
{
public:
	GpuJoinProcessor(int _iLeftWindowSize, int _iRightWindowSize);
	virtual ~GpuJoinProcessor();

	void Configure(GpuProcessor * _pPrevProcessor, GpuProcessorContext * _pContext, FILE * _fpLog);
	void Init(GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize);
	void Process(int _iNumEvents);
	void Print(FILE * _fp);
	GpuProcessor * Clone();
	int GetResultEventBufferIndex();
	char * GetResultEventBuffer();
	int GetResultEventBufferSize();

	void Print() { Print(stdout); }

	int GetLeftStreamWindowSize() { return i_LeftStraemWindowSize; }
	int GetRightStreamWindowSize() { return i_RightStraemWindowSize; }

private:
	int i_LeftStraemWindowSize;
	int i_RightStraemWindowSize;
	GpuProcessorContext * p_Context;
	GpuJoinKernel * p_JoinKernel;
	GpuProcessor * p_PrevProcessor;
};

}


#endif /* GPUJOINPROCESSOR_H_ */
