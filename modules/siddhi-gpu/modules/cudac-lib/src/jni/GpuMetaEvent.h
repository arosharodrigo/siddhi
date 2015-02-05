/*
 * GpuMetaEvent.h
 *
 *  Created on: Jan 20, 2015
 *      Author: prabodha
 */

#ifndef GPUMETAEVENT_H_
#define GPUMETAEVENT_H_


namespace SiddhiGpu
{

class GpuMetaAttribute
{
public:
	GpuMetaAttribute() :
		i_Type(-1),
		i_Length(-1),
		i_Position(-1)
	{}

	GpuMetaAttribute(int _iType, int _iLen, int _iPos) :
		i_Type(_iType),
		i_Length(_iLen),
		i_Position(_iPos)
	{}

	int i_Type;
	int i_Length;
	int i_Position;
};

class GpuMetaEvent
{
public:
	GpuMetaEvent(int _iStreamIndex, int _iAttribCount, int _iEventSize) :
		i_StreamIndex(_iStreamIndex),
		i_AttributeCount(_iAttribCount),
		i_SizeOfEventInBytes(_iEventSize)
	{
		p_Attributes = new GpuMetaAttribute[_iAttribCount];
	}

	~GpuMetaEvent()
	{
		delete [] p_Attributes;
	}

	GpuMetaEvent * Clone()
	{
		GpuMetaEvent * pCloned = new GpuMetaEvent(i_StreamIndex, i_AttributeCount, i_SizeOfEventInBytes);
		for(int i=0; i<i_AttributeCount; ++i)
		{
			pCloned->SetAttribute(i, p_Attributes[i].i_Type, p_Attributes[i].i_Length, p_Attributes[i].i_Position);
		}
		return pCloned;
	}

	void SetAttribute(int _iIndex, int _iType, int _iLen, int _iPos)
	{
		if(_iIndex >= 0 && _iIndex < i_AttributeCount)
		{
			p_Attributes[_iIndex].i_Type = _iType;
			p_Attributes[_iIndex].i_Position = _iPos;
			p_Attributes[_iIndex].i_Length = _iLen;
		}
	}

	int i_StreamIndex;
	int i_AttributeCount;
	int i_SizeOfEventInBytes;
	GpuMetaAttribute * p_Attributes;
};

};


#endif /* GPUMETAEVENT_H_ */
