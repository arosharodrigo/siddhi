package org.wso2.siddhi.core.gpu.query.selector;

import java.nio.ByteBuffer;
import java.util.List;

import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.ComplexEvent.Type;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEventPool;
import org.wso2.siddhi.core.event.stream.converter.StreamEventConverter;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor;

public class GpuQuerySelectorWorker implements Runnable {

    protected String workerId;
    
    protected ByteBuffer outputEventBuffer;
    protected int bufferStartPosition;
    protected int eventCount;
    
    protected List<AttributeProcessor> attributeProcessorList;
    protected List<GpuEventAttribute> gpuMetaEventAttributeList;
    protected StreamEventPool streamEventPool;
    
    protected StreamEvent firstEvent;
    protected StreamEvent lastEvent;
    
    protected ComplexEvent.Type eventTypes[]; 
    protected Object attributeData[];
    protected byte preAllocatedByteArray[];
    
    protected GpuMetaStreamEvent gpuMetaStreamEvent;
    protected StreamEventConverter streamEventConverter;
    
    protected int processedEventCount;
    
    public GpuQuerySelectorWorker(String id, StreamEventPool streamEventPool, StreamEventConverter streamEventConverter) {
        this.workerId = id;
        this.streamEventPool = streamEventPool;
        this.streamEventConverter = streamEventConverter;
        this.attributeProcessorList = null;

        this.gpuMetaEventAttributeList = null;
        this.gpuMetaStreamEvent = null;
        
        this.outputEventBuffer = null;
        this.bufferStartPosition = -1;
        this.eventCount = 0;
        
        this.firstEvent = null;
        this.lastEvent = null;
        this.attributeData = null;
        this.preAllocatedByteArray = null;
        
        this.eventTypes = ComplexEvent.Type.values();
        
        this.processedEventCount = 0;
    }

    public List<AttributeProcessor> getAttributeProcessorList() {
        return attributeProcessorList;
    }

    public void setAttributeProcessorList(List<AttributeProcessor> attributeProcessorList) {
        this.attributeProcessorList = attributeProcessorList;
    }

    @Override
    public void run() {
        processedEventCount = 0;
        for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) {

            ComplexEvent.Type type = eventTypes[outputEventBuffer.getShort()];

            if(type != Type.NONE) {
                StreamEvent borrowedEvent = streamEventPool.borrowEvent();

                long sequence = outputEventBuffer.getLong();
                long timestamp = outputEventBuffer.getLong();

                int index = 0;
                for (GpuEventAttribute attrib : gpuMetaEventAttributeList) {
                    switch(attrib.type) {
                    case BOOL:
                        attributeData[index++] = outputEventBuffer.getShort();
                        break;
                    case INT:
                        attributeData[index++] = outputEventBuffer.getInt();
                        break;
                    case LONG:
                        attributeData[index++] = outputEventBuffer.getLong();
                        break;
                    case FLOAT:
                        attributeData[index++] = outputEventBuffer.getFloat();
                        break;
                    case DOUBLE:
                        attributeData[index++] = outputEventBuffer.getDouble();
                        break;
                    case STRING:
                        short length = outputEventBuffer.getShort();
                        outputEventBuffer.get(preAllocatedByteArray, 0, attrib.length);
                        attributeData[index++] = new String(preAllocatedByteArray, 0, length).intern(); // TODO: avoid allocation
                        break;
                    }
                }

                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent);
                //log.debug("Converted event " + borrowedEvent.toString());

                // call actual select operations
                for (AttributeProcessor attributeProcessor : attributeProcessorList) {
                    attributeProcessor.process(borrowedEvent);
                }

                // add event to current list
                if (firstEvent == null) {
                    firstEvent = borrowedEvent;
                    lastEvent = firstEvent;
                } else {
                    lastEvent.setNext(borrowedEvent);
                    lastEvent = borrowedEvent;
                }
                
                processedEventCount++;

            } else {
                outputEventBuffer.position(outputEventBuffer.position() + gpuMetaStreamEvent.getEventSizeInBytes() - 2);
            }

        }
    }
    
    
    public void setOutputEventBuffer(ByteBuffer outputEventBuffer) {
        this.outputEventBuffer = outputEventBuffer;
    }
    
    public int getBufferStartPosition() {
        return bufferStartPosition;
    }

    public void setBufferStartPosition(int bufferStartPosition) {
        this.bufferStartPosition = bufferStartPosition;
        this.outputEventBuffer.position(bufferStartPosition);
    }

    public int getEventCount() {
        return eventCount;
    }

    public void setEventCount(int eventCount) {
        this.eventCount = eventCount;
    }

    public StreamEvent getFirstEvent() {
        return firstEvent;
    }
    
    public StreamEvent getLastEvent() {
        return lastEvent;
    }
    
    public int getProcessedEventCount() {
        return processedEventCount;
    }

    public void setGpuMetaStreamEvent(GpuMetaStreamEvent gpuMetaStreamEvent) {
        this.gpuMetaStreamEvent = gpuMetaStreamEvent;
        this.gpuMetaEventAttributeList = gpuMetaStreamEvent.getAttributes();
        
        int maxStringLength = 0;
        
        attributeData = new Object[gpuMetaEventAttributeList.size()];
        int index = 0;
        for (GpuEventAttribute attrib : gpuMetaEventAttributeList) {
            switch(attrib.type) {
            case BOOL:
                attributeData[index++] = new Boolean(false);
                break;
            case INT:
                attributeData[index++] = new Integer(0);
                break;
            case LONG:
                attributeData[index++] = new Long(0);
                break;
            case FLOAT:
                attributeData[index++] = new Float(0);
                break;
            case DOUBLE:
                attributeData[index++] = new Double(0);
                break;
            case STRING:
                attributeData[index++] = new String();
                maxStringLength = (attrib.length > maxStringLength ? attrib.length : maxStringLength);
                break;
            }
        }
        
        preAllocatedByteArray = new byte[maxStringLength + 1];
    }
    
    public void setAttributeData(int index, int value) {
        attributeData[index] = value;
    }
    
    public void setAttributeData(int index, long value) {
        attributeData[index] = value;
    }
    
    public void setAttributeData(int index, float value) {
        attributeData[index] = value;
    }
    
    public void setAttributeData(int index, double value) {
        attributeData[index] = value;
    }
    
    public void setAttributeData(int index, short value) {
        attributeData[index] = value;
    }
    
    public void setAttributeData(int index, String value) {
        attributeData[index] = value;
    }
}
