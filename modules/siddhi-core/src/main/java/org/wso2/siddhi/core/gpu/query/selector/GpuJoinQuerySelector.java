package org.wso2.siddhi.core.gpu.query.selector;

import org.wso2.siddhi.core.config.ExecutionPlanContext;
import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.ComplexEvent.Type;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor;
import org.wso2.siddhi.query.api.execution.query.selection.Selector;

public class GpuJoinQuerySelector extends GpuQuerySelector {

    private int segmentEventCount;
    
    public GpuJoinQuerySelector(String id, Selector selector, boolean currentOn, boolean expiredOn, ExecutionPlanContext executionPlanContext) {
        super(id, selector, currentOn, expiredOn, executionPlanContext);
    }

    @Override
    public void process(int eventCount) {
        outputEventBuffer.position(0);
        inputEventBuffer.position(0);
        
      //log.debug("<" + eventCount + "> Converting eventCount=" + eventCount + " eventSegmentSize=" + eventSegmentSize);
        int indexInsideSegment = 0;
        int segIdx = 0;
        for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) {

            segIdx = resultsIndex / segmentEventCount;
            
            StreamEvent borrowedEvent = streamEventPool.borrowEvent();

            ComplexEvent.Type type = eventTypes[outputEventBuffer.getShort()];
            
//            log.debug("<" + eventCount + "> Converting eventIndex=" + resultsIndex + " type=" + type 
//                    + " segIdx=" + segIdx + " segInternalIdx=" + indexInsideSegment);
            
            if(type != Type.NONE && type != Type.RESET) {
                long sequence = outputEventBuffer.getLong();
                long timestamp = outputEventBuffer.getLong();

                int index = 0;
                for (GpuEventAttribute attrib : gpuMetaStreamEvent.getAttributes()) {
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
                        attributeData[index++] = new String(preAllocatedByteArray, 0, length); // TODO: avoid allocation
                        break;
                    }
                }

                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent);
//                log.debug("<" + eventCount + "> Converted event " + resultsIndex + " : " + borrowedEvent.toString());
                
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
                
                indexInsideSegment++;
                indexInsideSegment = indexInsideSegment % segmentEventCount;

            } else if (type == Type.RESET){
                // skip remaining bytes in segment
//                log.debug("<" + eventCount + "> Skip to next segment : CurrPos=" + eventBuffer.position() + " SegIdx=" + indexInsideSegment + 
//                        " EventSize=" + gpuMetaStreamEvent.getEventSizeInBytes());
                
                outputEventBuffer.position(
                        outputEventBuffer.position() + 
                        ((segmentEventCount - indexInsideSegment) * gpuMetaStreamEvent.getEventSizeInBytes()) 
                        - 2);
                
//                log.debug("<" + eventCount + "> buffer new pos : " + eventBuffer.position());
                resultsIndex = ((segIdx + 1) * segmentEventCount) - 1;
                indexInsideSegment = 0;
            }
        }
        
        // call output rate limiter
        if (firstEvent != null) {
            outputComplexEventChunk.add(firstEvent);
            outputRateLimiter.process(outputComplexEventChunk);
        }
        firstEvent = null;
        lastEvent = null;
        outputComplexEventChunk.clear();
    }

    public int getSegmentEventCount() {
        return segmentEventCount;
    }

    public void setSegmentEventCount(int segmentEventCount) {
        this.segmentEventCount = segmentEventCount;
    }
    
}
