package org.wso2.siddhi.core.gpu.query.selector;

import java.util.List;

import org.apache.log4j.Logger;
import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.ComplexEvent.Type;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEventPool;
import org.wso2.siddhi.core.event.stream.converter.StreamEventConverter;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.query.selector.attribute.processor.AttributeProcessor;

public class GpuJoinQuerySelectorWorker extends GpuQuerySelectorWorker {
    
    private static final Logger log = Logger.getLogger(GpuJoinQuerySelectorWorker.class);
    protected int segmentEventCount;
    protected int workStartEvent;
    protected int workEndEvent;
    
    public GpuJoinQuerySelectorWorker(String id, StreamEventPool streamEventPool, StreamEventConverter streamEventConverter) {
        super(id, streamEventPool, streamEventConverter);
        
        this.segmentEventCount = 0;
        this.setWorkStartEvent(0);
        this.setWorkEndEvent(0);
    }
    
    @Override
    public void run() {
        processedEventCount = 0;
        int indexInsideSegment = 0;
        int segIdx = 0;
        ComplexEvent.Type type;
        
//        log.debug("<GpuJoinQuerySelectorWorker_" + workerId + "> process start=" + workStartEvent + " end=" + workEndEvent 
//                + " segmentEventCount=" + segmentEventCount 
//                + " eventSize=" + gpuMetaStreamEvent.getEventSizeInBytes()
//                + " eventBuffer=" + outputEventBuffer);
//        
            
        for (int resultsIndex = workStartEvent; resultsIndex < workEndEvent; ++resultsIndex) {

            segIdx = resultsIndex / segmentEventCount;

            type = eventTypes[outputEventBuffer.getShort()]; // 1 -> 2 bytes

//            log.debug("<GpuJoinQuerySelectorWorker_" + workerId + "> process eventIndex=" + resultsIndex + " type=" + type 
//                    + " segIdx=" + segIdx + " segInternalIdx=" + indexInsideSegment);

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
                        attributeData[index++] = new String(preAllocatedByteArray, 0, length).intern(); // TODO: avoid allocation
                        break;
                    }
                }

                //XXX: assume always ZeroStreamEventConvertor
                //                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent); 
                System.arraycopy(attributeData, 0, borrowedEvent.getOutputData(), 0, index);

//                log.debug("<GpuJoinQuerySelectorWorker_" + workerId + "> Converted event " + resultsIndex + " : [" + sequence + "] " + borrowedEvent.toString());

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
                indexInsideSegment++;
                indexInsideSegment = indexInsideSegment % segmentEventCount;

            } else if (type == Type.RESET){
                // skip remaining bytes in segment
//                log.debug("<GpuJoinQuerySelectorWorker_" + workerId + "> Skip to next segment : CurrPos=" + 
//                        outputEventBuffer.position() + " segInternalIdx=" + indexInsideSegment);

                outputEventBuffer.position(
                        outputEventBuffer.position() + 
                        ((segmentEventCount - indexInsideSegment) * gpuMetaStreamEvent.getEventSizeInBytes()) 
                        - 2);

//                log.debug("<GpuJoinQuerySelectorWorker_" + workerId + "> buffer new pos : " + outputEventBuffer.position());
                resultsIndex = ((segIdx + 1) * segmentEventCount) - 1;
                indexInsideSegment = 0;
            }
        }
    }

    public int getSegmentEventCount() {
        return segmentEventCount;
    }

    public void setSegmentEventCount(int segmentEventCount) {
        this.segmentEventCount = segmentEventCount;
    }

    public int getWorkStartEvent() {
        return workStartEvent;
    }

    public void setWorkStartEvent(int workStartEvent) {
        this.workStartEvent = workStartEvent;
    }

    public int getWorkEndEvent() {
        return workEndEvent;
    }

    public void setWorkEndEvent(int workEndEvent) {
        this.workEndEvent = workEndEvent;
    }
}
