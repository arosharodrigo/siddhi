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
class ExecutorNode;

class GpuJoinProcessor : public GpuProcessor
{
public:
	GpuJoinProcessor(int _iLeftWindowSize, int _iRightWindowSize);
	virtual ~GpuJoinProcessor();

	void Configure(int _iStreamIndex, GpuProcessor * _pPrevProcessor, GpuProcessorContext * _pContext, FILE * _fpLog);
	void Init(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize);
	int Process(int _iStreamIndex, int _iNumEvents);
	void Print(FILE * _fp);
	GpuProcessor * Clone();
	int GetResultEventBufferIndex();
	char * GetResultEventBuffer();
	int GetResultEventBufferSize();

	char * GetLeftResultEventBuffer();
	int GetLeftResultEventBufferSize();
	char * GetRightResultEventBuffer();
	int GetRightResultEventBufferSize();

	void Print() { Print(stdout); }

	void SetLeftStreamWindowSize(int _iWindowSize) { i_LeftStraemWindowSize = _iWindowSize; }
	void SetRightStreamWindowSize(int _iWindowSize) { i_RightStraemWindowSize = _iWindowSize; }
	int GetLeftStreamWindowSize() { return i_LeftStraemWindowSize; }
	int GetRightStreamWindowSize() { return i_RightStraemWindowSize; }

	void SetLeftTrigger(bool _bTrigger) { b_LeftTrigger = _bTrigger; }
	void SetRightTrigger(bool _bTrigger) { b_RightTrigger = _bTrigger; }
	bool GetLeftTrigger() { return b_LeftTrigger; }
	bool GetRightTrigger() { return b_RightTrigger; }

	void SetWithInTimeMilliSeconds(uint64_t _iTime) { i_WithInTimeMilliSeconds = _iTime; }
	uint64_t GetWithInTimeMilliSeconds() { return i_WithInTimeMilliSeconds; }

	void SetExecutorNodes(int _iNodeCount);
	void AddExecutorNode(int _iPos, ExecutorNode & _pNode);

	void SetThreadWorkSize(int _iSize) { i_ThreadWorkSize = _iSize; }
	int GetThreadWorkSize() { return i_ThreadWorkSize; }

	int            i_NodeCount;
	ExecutorNode * ap_ExecutorNodes; // nodes are stored in in-order

private:
	int i_LeftStraemWindowSize;
	int i_RightStraemWindowSize;

	GpuMetaEvent * p_LeftStreamMetaEvent;
	GpuMetaEvent * p_RightStreamMetaEvent;

	bool b_LeftTrigger;
	bool b_RightTrigger;

	uint64_t i_WithInTimeMilliSeconds;
	int i_ThreadWorkSize;

	GpuProcessorContext * p_LeftContext;
	GpuProcessorContext * p_RightContext;

	GpuJoinKernel * p_JoinKernel;
	GpuProcessor * p_LeftPrevProcessor;
	GpuProcessor * p_RightPrevProcessor;

	FILE * fp_LeftLog;
	FILE * fp_RightLog;
};

}


#endif /* GPUJOINPROCESSOR_H_ */
