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
#include "DataTypes.h"
#include "GpuMetaEvent.h"

namespace SiddhiGpu
{

class GpuKernel;
class GpuProcessorContext;

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

	GpuProcessor(Type _eType) :
		e_Type(_eType),
		p_Next(NULL),
		i_ThreadBlockSize(128),
		p_OutputStreamMeta(NULL),
		p_OutputAttributeMapping(NULL),
		b_CurrentOn(true),
		b_ExpiredOn(true),
		fp_Log(NULL)
	{}

	virtual ~GpuProcessor() {}

	virtual void Configure(int _iStreamIndex, GpuProcessor * _pPrevProcessor, GpuProcessorContext * _pContext, FILE * _fpLog) = 0;
	virtual void Init(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize) = 0;
	virtual int Process(int _iStreamIndex, int _iNumEvents) = 0;
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

	void SetOutputStream(GpuMetaEvent * _pOutputStreamMeta, AttributeMappings * _pMappings)
	{
		p_OutputStreamMeta = _pOutputStreamMeta->Clone();
		p_OutputAttributeMapping = _pMappings->Clone();
	}

	void SetCurrentOn(bool _on) { b_CurrentOn = _on; }
	void SetExpiredOn(bool _on) { b_ExpiredOn = _on; }

	bool GetCurrentOn() { return b_CurrentOn; }
	bool GetExpiredOn() { return b_ExpiredOn; }

protected:
	Type e_Type;
	GpuProcessor * p_Next;
	int i_ThreadBlockSize;
	GpuMetaEvent * p_OutputStreamMeta;
	AttributeMappings * p_OutputAttributeMapping;
	bool b_CurrentOn;
	bool b_ExpiredOn;
	FILE * fp_Log;
};

};



#endif /* GPUPROCESSOR_H_ */
