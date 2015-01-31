/*
 * GpuProcessor.h
 *
 *  Created on: Jan 19, 2015
 *      Author: prabodha
 */

#ifndef GPUPROCESSOR_H_
#define GPUPROCESSOR_H_

#include <stdlib.h>
#include <stdio.h>
#include <list>

namespace SiddhiGpu
{

class GpuKernel;
class GpuProcessorContext;
class GpuMetaEvent;

class GpuProcessor
{
public:
	enum Type
	{
		FILTER = 0,
		LENGTH_SLIDING_WINDOW,
		LENGTH_BATCH_WINDOW,
		TIME_SLIDING_WINDOW,
		TIME_BATCH_WINDOW,
		JOIN,
		SEQUENCE,
		PATTERN
	};

	GpuProcessor(Type _eType) : e_Type(_eType), p_Next(NULL), i_ThreadBlockSize(128), fp_Log(NULL) {}
	virtual ~GpuProcessor() {}

	virtual void Configure(GpuProcessor * _pPrevProcessor, GpuProcessorContext * _pContext, FILE * _fpLog) = 0;
	virtual void Init(GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize) = 0;
	virtual void Process(int _iNumEvents) = 0;
	virtual void Print(FILE * _fp) = 0;
	virtual GpuProcessor * Clone() = 0;

	virtual int GetResultEventBufferIndex() = 0;
	virtual char * GetResultEventBuffer() = 0;
	virtual int GetResultEventBufferSize() = 0;

	Type GetType() { return e_Type; }
	GpuProcessor * GetNext() { return p_Next; }
	void SetNext(GpuProcessor * _pProc) { p_Next = _pProc; }
	void AddToLast(GpuProcessor * _pProc) { if(p_Next) p_Next->AddToLast(_pProc); else p_Next = _pProc; }
	void SetThreadBlockSize(int _iThreadBlockSize) { i_ThreadBlockSize = _iThreadBlockSize; }

protected:
	Type e_Type;
	GpuProcessor * p_Next;
	std::list<GpuKernel*> lst_Kernels;
	int i_ThreadBlockSize;
	FILE * fp_Log;
};

};



#endif /* GPUPROCESSOR_H_ */
