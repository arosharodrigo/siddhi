/*
 * GpuKernelDataTypes.h
 *
 *  Created on: Jan 24, 2015
 *      Author: prabodha
 */

#ifndef GPUKERNELDATATYPES_H_
#define GPUKERNELDATATYPES_H_

#include <stdint.h>

namespace SiddhiGpu
{

class ExecutorNode;

#pragma pack(1)

typedef struct GpuKernelMetaAttribute
{
	int i_Type;
	int i_Length;
	int i_Position;
}GpuKernelMetaAttribute;

typedef struct GpuKernelMetaEvent
{
	int i_StreamIndex;
	int i_AttributeCount;
	int i_SizeOfEventInBytes;
	GpuKernelMetaAttribute p_Attributes[1];
}GpuKernelMetaEvent;


typedef struct GpuKernelFilter
{
	int            i_NodeCount;
	ExecutorNode * ap_ExecutorNodes; // nodes are stored in in-order
}GpuKernelFilter;

typedef struct GpuEvent
{
	enum Type
	{
		CURRENT = 0,
		EXPIRED,
		TIMER,
		RESET,
		NONE
	};

	uint16_t i_Type; // 2 bytes
	uint64_t i_Sequence; // 8 bytes
	uint64_t i_Timestamp; // 8 bytes
	char a_Attributes[1]; // n bytes
} GpuEvent;

#pragma pack()

}

#endif /* GPUKERNELDATATYPES_H_ */
