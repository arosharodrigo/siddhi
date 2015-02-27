package org.wso2.siddhi.core.gpu.query.selector;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

import org.wso2.siddhi.core.config.ExecutionPlanContext;
import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor;
import org.wso2.siddhi.query.api.execution.query.selection.Selector;

public class GpuFilterQuerySelector extends GpuQuerySelector {

    private IntBuffer outputEventIndexBuffer;
    
    public GpuFilterQuerySelector(String id, Selector selector, boolean currentOn, boolean expiredOn, ExecutionPlanContext executionPlanContext) {
        super(id, selector, currentOn, expiredOn, executionPlanContext);
    }
    
    protected void deserialize(int eventCount) {
        for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) {
            int matched = outputEventIndexBuffer.get();
            if (matched >= 0) {

                StreamEvent borrowedEvent = streamEventPool.borrowEvent();

                ComplexEvent.Type type = eventTypes[inputEventBuffer.getShort()];
                long sequence = inputEventBuffer.getLong();
                long timestamp = inputEventBuffer.getLong();

                //                log.debug("matched event : index=" + resultsIndex + " type=" + type + " seq=" + sequence + " time=" + timestamp);

                int index = 0;
                for (GpuEventAttribute attrib : gpuMetaEventAttributeList) {
                    switch(attrib.type) {
                    case BOOL:
                        attributeData[index++] = inputEventBuffer.getShort();
                        break;
                    case INT:
                        attributeData[index++] = inputEventBuffer.getInt();
                        break;
                    case LONG:
                        attributeData[index++] = inputEventBuffer.getLong();
                        break;
                    case FLOAT:
                        attributeData[index++] = inputEventBuffer.getFloat();
                        break;
                    case DOUBLE:
                        attributeData[index++] = inputEventBuffer.getDouble();
                        break;
                    case STRING:
                        short length = inputEventBuffer.getShort();
                        inputEventBuffer.get(preAllocatedByteArray, 0, attrib.length);
                        attributeData[index++] = new String(preAllocatedByteArray, 0, length); // TODO: avoid allocation
                        break;
                    }
                }

                //                for(int i=0; i<index; ++i) {
                //                    log.debug("attributeData: " + attributeData[i]);
                //                }

                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent);

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

            } else {
                //                log.debug("not matched event : index=" + resultsIndex + " bufferPosition=" + inputEventBuffer.position());
                inputEventBuffer.position(inputEventBuffer.position() + gpuMetaStreamEvent.getEventSizeInBytes());
            }
        }
    }
    
    @Override
    public void process(int eventCount) {
        outputEventIndexBuffer.position(0);
        inputEventBuffer.position(0);
        
//        for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) {
//            int matched = outputEventIndexBuffer.get();
//            if (matched >= 0) {
//
//                StreamEvent borrowedEvent = streamEventPool.borrowEvent();
//
//                ComplexEvent.Type type = eventTypes[inputEventBuffer.getShort()];
//                long sequence = inputEventBuffer.getLong();
//                long timestamp = inputEventBuffer.getLong();
//
//                //                log.debug("matched event : index=" + resultsIndex + " type=" + type + " seq=" + sequence + " time=" + timestamp);
//
//                int index = 0;
//                for (GpuEventAttribute attrib : gpuMetaEventAttributeList) {
//                    switch(attrib.type) {
//                    case BOOL:
//                        attributeData[index++] = inputEventBuffer.getShort();
//                        break;
//                    case INT:
//                        attributeData[index++] = inputEventBuffer.getInt();
//                        break;
//                    case LONG:
//                        attributeData[index++] = inputEventBuffer.getLong();
//                        break;
//                    case FLOAT:
//                        attributeData[index++] = inputEventBuffer.getFloat();
//                        break;
//                    case DOUBLE:
//                        attributeData[index++] = inputEventBuffer.getDouble();
//                        break;
//                    case STRING:
//                        short length = inputEventBuffer.getShort();
//                        inputEventBuffer.get(preAllocatedByteArray, 0, attrib.length);
//                        attributeData[index++] = new String(preAllocatedByteArray, 0, length); // TODO: avoid allocation
//                        break;
//                    }
//                }
//
//                //                for(int i=0; i<index; ++i) {
//                //                    log.debug("attributeData: " + attributeData[i]);
//                //                }
//
//                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent);
//                //log.debug("Converted event " + borrowedEvent.toString());
//
//                // call actual select operations
//                for (AttributeProcessor attributeProcessor : attributeProcessorList) {
//                    attributeProcessor.process(borrowedEvent);
//                }
//                
//                // add event to current list
//                if (firstEvent == null) {
//                    firstEvent = borrowedEvent;
//                    lastEvent = firstEvent;
//                } else {
//                    lastEvent.setNext(borrowedEvent);
//                    lastEvent = borrowedEvent;
//                }
//                
//            } else {
//                //                log.debug("not matched event : index=" + resultsIndex + " bufferPosition=" + inputEventBuffer.position());
//                inputEventBuffer.position(inputEventBuffer.position() + gpuMetaStreamEvent.getEventSizeInBytes());
//            }
//        }
        
        deserialize(eventCount);
        
        if (firstEvent != null) {
            outputComplexEventChunk.add(firstEvent);
            outputRateLimiter.process(outputComplexEventChunk);
        }
        
        firstEvent = null;
        lastEvent = null;
        outputComplexEventChunk.clear();
    }
    
    @Override
    public void setOutputEventBuffer(ByteBuffer outputEventBuffer) {
        this.outputEventBuffer = outputEventBuffer;
        this.outputEventIndexBuffer = outputEventBuffer.asIntBuffer();
    }
}
