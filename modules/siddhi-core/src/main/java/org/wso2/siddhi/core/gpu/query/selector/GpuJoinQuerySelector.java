package org.wso2.siddhi.core.gpu.query.selector;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.wso2.siddhi.core.config.ExecutionPlanContext;
import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.ComplexEvent.Type;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.query.selector.QuerySelector;
import org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor;
import org.wso2.siddhi.query.api.execution.query.selection.Selector;

public class GpuJoinQuerySelector extends GpuQuerySelector {
    private static final Logger log = Logger.getLogger(GpuJoinQuerySelector.class);
    private int segmentEventCount;
    private int threadWorkSize;
    
    public GpuJoinQuerySelector(String id, Selector selector, boolean currentOn, boolean expiredOn, ExecutionPlanContext executionPlanContext) {
        super(id, selector, currentOn, expiredOn, executionPlanContext);
        this.segmentEventCount = -1;
        this.setThreadWorkSize(0);
    }
  
    @Override
    public void process(int eventCount) {
        outputEventBuffer.position(0);
        inputEventBuffer.position(0);

        log.debug("<" + id + " @ GpuJoinQuerySelector> process eventCount=" + eventCount + " eventSegmentSize=" + segmentEventCount);

        int indexInsideSegment = 0;
        int segIdx = 0;
        for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) {

            segIdx = resultsIndex / segmentEventCount;

            ComplexEvent.Type type = eventTypes[outputEventBuffer.getShort()]; // 1 -> 2 bytes

            log.debug("<" + id + " @ GpuJoinQuerySelector> process eventIndex=" + resultsIndex + " type=" + type 
                    + " segIdx=" + segIdx + " segInternalIdx=" + indexInsideSegment);

            if(type != Type.NONE && type != Type.RESET) {
                StreamEvent borrowedEvent = streamEventPool.borrowEvent();
                borrowedEvent.setType(type);
                
                long sequence = outputEventBuffer.getLong(); // 2 -> 8 bytes
                borrowedEvent.setTimestamp(outputEventBuffer.getLong()); // 3 -> 8bytes

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
                        attributeData[index++] = new String(preAllocatedByteArray, 0, length); // TODO: avoid allocation
                        break;
                    }
                }
                
                //XXX: assume always ZeroStreamEventConvertor
                //                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent); 
                System.arraycopy(attributeData, 0, borrowedEvent.getOutputData(), 0, index);

                log.debug("<" + id + " @ GpuJoinQuerySelector> Converted event " + resultsIndex + " : [" + sequence + "] " + borrowedEvent.toString());

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
                log.debug("<" + id + " @ GpuJoinQuerySelector> Skip to next segment : CurrPos=" + 
                        outputEventBuffer.position() + " segInternalIdx=" + indexInsideSegment);

                outputEventBuffer.position(
                        outputEventBuffer.position() + 
                        ((segmentEventCount - indexInsideSegment) * gpuMetaStreamEvent.getEventSizeInBytes()) 
                        - 2);

                log.debug("<" + id + " @ GpuJoinQuerySelector> buffer new pos : " + outputEventBuffer.position());
                resultsIndex = ((segIdx + 1) * segmentEventCount) - 1;
                indexInsideSegment = 0;
            }
        }

        //        log.debug("<" + id + " @ GpuJoinQuerySelector> Call outputRateLimiter " + outputRateLimiter);

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
    
    @Override
    public QuerySelector clone(String key) {
        GpuJoinQuerySelector clonedQuerySelector = new GpuJoinQuerySelector(id + key, selector, currentOn, expiredOn, executionPlanContext);
        List<AttributeProcessor> clonedAttributeProcessorList = new ArrayList<AttributeProcessor>();
        for (AttributeProcessor attributeProcessor : attributeProcessorList) {
            clonedAttributeProcessorList.add(attributeProcessor.cloneProcessor());
        }
        clonedQuerySelector.attributeProcessorList = clonedAttributeProcessorList;
        clonedQuerySelector.eventPopulator = eventPopulator;
        clonedQuerySelector.segmentEventCount = segmentEventCount;
        clonedQuerySelector.outputRateLimiter = outputRateLimiter;
        return clonedQuerySelector;
    }

    public int getThreadWorkSize() {
        return threadWorkSize;
    }

    public void setThreadWorkSize(int threadWorkSize) {
        this.threadWorkSize = threadWorkSize;
        if(threadWorkSize != 0) {
            segmentEventCount = threadWorkSize;
        }
    }
}
