/*
 * ByteBufferStructs.h
 *
 *  Created on: Nov 9, 2014
 *      Author: prabodha
 */

#ifndef BYTEBUFFERSTRUCTS_H_
#define BYTEBUFFERSTRUCTS_H_

// byte aligned data structures
#pragma pack(1)

typedef struct EventAttributeMeta
{
	uint16_t i_Type; // 2 bytes
	uint16_t i_Length; // 2 bytes
} EventAttributeMeta;

typedef struct EventMeta
{
	uint16_t i_AttributeCount; // 2 bytes
	EventAttributeMeta a_Attributes[1]; // array of attributes
} EventMeta;

typedef struct MatchedEvents
{
	uint32_t i_ResultCount; // 4 bytes
	uint32_t a_ResultEvents[1]; // array of ints
} MatchedEvents;

#pragma pack()


#endif /* BYTEBUFFERSTRUCTS_H_ */
