package org.wso2.siddhi.core.gpu.event.stream.converter;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

import org.apache.log4j.Logger;
import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.ComplexEvent.Type;
import org.wso2.siddhi.core.event.ComplexEventChunk;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.event.stream.MetaStreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEventPool;
import org.wso2.siddhi.core.event.stream.converter.ConversionStreamEventChunk;
import org.wso2.siddhi.core.event.stream.converter.StreamEventConverter;
import org.wso2.siddhi.core.event.stream.converter.StreamEventConverterFactory;
import org.wso2.siddhi.core.gpu.event.stream.GpuEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuEventPool;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;

public class ConversionGpuEventChunk extends ConversionStreamEventChunk {

    private static final Logger log = Logger.getLogger(ConversionGpuEventChunk.class);
    private GpuMetaStreamEvent gpuMetaStreamEvent;
    private Object attributeData[];
    private ComplexEvent.Type eventTypes[]; 
    
    public ConversionGpuEventChunk(MetaStreamEvent metaStreamEvent, StreamEventPool streamEventPool, GpuMetaStreamEvent gpuMetaStreamEvent) {
        super(metaStreamEvent, streamEventPool);
        this.gpuMetaStreamEvent = gpuMetaStreamEvent;
        eventTypes = ComplexEvent.Type.values();
        
        attributeData = new Object[gpuMetaStreamEvent.getAttributes().size()];
        int index = 0;
        for (GpuEventAttribute attrib : gpuMetaStreamEvent.getAttributes()) {
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
                break;
            }
        }
    }

    public ConversionGpuEventChunk(StreamEventConverter streamEventConverter, StreamEventPool streamEventPool, GpuMetaStreamEvent gpuMetaStreamEvent) {
        super(streamEventConverter, streamEventPool);
        this.gpuMetaStreamEvent = gpuMetaStreamEvent;
    }
    
    public void convertAndAdd(ByteBuffer eventBuffer, int eventCount) {
        for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) {

            StreamEvent borrowedEvent = streamEventPool.borrowEvent();

            ComplexEvent.Type type = eventTypes[eventBuffer.getShort()];
            
            if(type != Type.NONE) {
                long sequence = eventBuffer.getLong();
                long timestamp = eventBuffer.getLong();

                int index = 0;
                for (GpuEventAttribute attrib : gpuMetaStreamEvent.getAttributes()) {
                    switch(attrib.type) {
                    case BOOL:
                        attributeData[index++] = eventBuffer.getShort();
                        break;
                    case INT:
                        attributeData[index++] = eventBuffer.getInt();
                        break;
                    case LONG:
                        attributeData[index++] = eventBuffer.getLong();
                        break;
                    case FLOAT:
                        attributeData[index++] = eventBuffer.getFloat();
                        break;
                    case DOUBLE:
                        attributeData[index++] = eventBuffer.getDouble();
                        break;
                    case STRING:
                        short length = eventBuffer.getShort();
                        byte string[] = new byte[attrib.length];
                        eventBuffer.get(string, 0, attrib.length);
                        attributeData[index++] = new String(string, 0, length); // TODO: avoid allocation
                        break;
                    }
                }

                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent);
                //log.debug("Converted event " + borrowedEvent.toString());

                if (first == null) {
                    first = borrowedEvent;
                    last = first;
                    currentEventCount = 1;
                } else {
                    last.setNext(borrowedEvent);
                    last = borrowedEvent;
                    currentEventCount++;
                }
                
            } else {
                eventBuffer.position(eventBuffer.position() + gpuMetaStreamEvent.getEventSizeInBytes() - 2);
            }
            
        }
    }
    
    public void convertAndAdd(ByteBuffer eventBuffer, int eventCount, int eventSegmentSize) {
        //log.debug("<" + eventCount + "> Converting eventCount=" + eventCount + " eventSegmentSize=" + eventSegmentSize);
        int indexInsideSegment = 0;
        int segIdx = 0;
        for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) {

            segIdx = resultsIndex / eventSegmentSize;
            
            StreamEvent borrowedEvent = streamEventPool.borrowEvent();

            ComplexEvent.Type type = eventTypes[eventBuffer.getShort()];
            
//            log.debug("<" + eventCount + "> Converting eventIndex=" + resultsIndex + " type=" + type 
//                    + " segIdx=" + segIdx + " segInternalIdx=" + indexInsideSegment);
            
            if(type != Type.NONE && type != Type.RESET) {
                long sequence = eventBuffer.getLong();
                long timestamp = eventBuffer.getLong();

                int index = 0;
                for (GpuEventAttribute attrib : gpuMetaStreamEvent.getAttributes()) {
                    switch(attrib.type) {
                    case BOOL:
                        attributeData[index++] = eventBuffer.getShort();
                        break;
                    case INT:
                        attributeData[index++] = eventBuffer.getInt();
                        break;
                    case LONG:
                        attributeData[index++] = eventBuffer.getLong();
                        break;
                    case FLOAT:
                        attributeData[index++] = eventBuffer.getFloat();
                        break;
                    case DOUBLE:
                        attributeData[index++] = eventBuffer.getDouble();
                        break;
                    case STRING:
                        short length = eventBuffer.getShort();
                        byte string[] = new byte[attrib.length];
                        eventBuffer.get(string, 0, attrib.length);
                        attributeData[index++] = new String(string, 0, length); // TODO: avoid allocation
                        break;
                    }
                }

                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent);
//                log.debug("<" + eventCount + "> Converted event " + resultsIndex + " : " + borrowedEvent.toString());

                if (first == null) {
                    first = borrowedEvent;
                    last = first;
                    currentEventCount = 1;
                } else {
                    last.setNext(borrowedEvent);
                    last = borrowedEvent;
                    currentEventCount++;
                }
                
                indexInsideSegment++;
                indexInsideSegment = indexInsideSegment % eventSegmentSize;

            } else if (type == Type.RESET){
                // skip remaining bytes in segment
//                log.debug("<" + eventCount + "> Skip to next segment : CurrPos=" + eventBuffer.position() + " SegIdx=" + indexInsideSegment + 
//                        " EventSize=" + gpuMetaStreamEvent.getEventSizeInBytes());
                
                eventBuffer.position(eventBuffer.position() + ((eventSegmentSize - indexInsideSegment) * gpuMetaStreamEvent.getEventSizeInBytes()) - 2);
                
//                log.debug("<" + eventCount + "> buffer new pos : " + eventBuffer.position());
                resultsIndex = ((segIdx + 1) * eventSegmentSize) - 1;
                indexInsideSegment = 0;
            }
        }
    }
    
    public void convertAndAdd(IntBuffer indexBuffer, ByteBuffer eventBuffer, int eventCount) {
        
        for (int resultsIndex = 0; resultsIndex < eventCount; ++resultsIndex) {
            int matched = indexBuffer.get();
            if (matched >= 0) {

                StreamEvent borrowedEvent = streamEventPool.borrowEvent();
                
                ComplexEvent.Type type = eventTypes[eventBuffer.getShort()];
                long sequence = eventBuffer.getLong();
                long timestamp = eventBuffer.getLong();
                
//                log.debug("matched event : index=" + resultsIndex + " type=" + type + " seq=" + sequence + " time=" + timestamp);
                
                int index = 0;
                for (GpuEventAttribute attrib : gpuMetaStreamEvent.getAttributes()) {
                    switch(attrib.type) {
                    case BOOL:
                        attributeData[index++] = eventBuffer.getShort();
                        break;
                    case INT:
                        attributeData[index++] = eventBuffer.getInt();
                        break;
                    case LONG:
                        attributeData[index++] = eventBuffer.getLong();
                        break;
                    case FLOAT:
                        attributeData[index++] = eventBuffer.getFloat();
                        break;
                    case DOUBLE:
                        attributeData[index++] = eventBuffer.getDouble();
                        break;
                    case STRING:
                        short length = eventBuffer.getShort();
                        byte string[] = new byte[attrib.length];
                        eventBuffer.get(string, 0, attrib.length);
                        attributeData[index++] = new String(string, 0, length); // TODO: avoid allocation
                        break;
                    }
                }
                
//                for(int i=0; i<index; ++i) {
//                    log.debug("attributeData: " + attributeData[i]);
//                }
                
                streamEventConverter.convertData(timestamp, type, attributeData, borrowedEvent);
                //log.debug("Converted event " + borrowedEvent.toString());
                
                if (first == null) {
                    first = borrowedEvent;
                    last = first;
                    currentEventCount = 1;
                } else {
                    last.setNext(borrowedEvent);
                    last = borrowedEvent;
                    currentEventCount++;
                }
            } else {
//                log.debug("not matched event : index=" + resultsIndex + " bufferPosition=" + eventBuffer.position());
                eventBuffer.position(eventBuffer.position() + gpuMetaStreamEvent.getEventSizeInBytes());
            }
        }
    }
    
    public void convertAndAssign(Event event) {
        StreamEvent borrowedEvent = streamEventPool.borrowEvent();
        streamEventConverter.convertEvent(event, borrowedEvent);
        first = borrowedEvent;
        last = first;
        currentEventCount = 1;
    }

    public void convertAndAssign(long timeStamp, Object[] data) {
        StreamEvent borrowedEvent = streamEventPool.borrowEvent();
        streamEventConverter.convertData(timeStamp, data, borrowedEvent);
        first = borrowedEvent;
        last = first;
        currentEventCount = 1;
    }

    public void convertAndAssign(ComplexEvent complexEvent) {
        first = streamEventPool.borrowEvent();
        currentEventCount = 1;
        last = convertAllStreamEvents(complexEvent, first);
    }

    public void convertAndAssign(Event[] events) {
        StreamEvent firstEvent = streamEventPool.borrowEvent();
        streamEventConverter.convertEvent(events[0], firstEvent);
        StreamEvent currentEvent = firstEvent;
        for (int i = 1, eventsLength = events.length; i < eventsLength; i++) {
            StreamEvent nextEvent = streamEventPool.borrowEvent();
            streamEventConverter.convertEvent(events[i], nextEvent);
            currentEvent.setNext(nextEvent);
            currentEvent = nextEvent;
        }
        first = firstEvent;
        last = currentEvent;
        currentEventCount = events.length;
    }

    public void convertAndAdd(Event event) {
        StreamEvent borrowedEvent = streamEventPool.borrowEvent();
        streamEventConverter.convertEvent(event, borrowedEvent);

        if (first == null) {
            first = borrowedEvent;
            last = first;
            currentEventCount = 1;
        } else {
            last.setNext(borrowedEvent);
            last = borrowedEvent;
            currentEventCount++;
        }

    }

    private StreamEvent convertAllStreamEvents(ComplexEvent complexEvents, StreamEvent firstEvent) {
        streamEventConverter.convertStreamEvent(complexEvents, firstEvent);
        StreamEvent currentEvent = firstEvent;
        complexEvents = complexEvents.getNext();
        while (complexEvents != null) {
            StreamEvent nextEvent = streamEventPool.borrowEvent();
            streamEventConverter.convertStreamEvent(complexEvents, nextEvent);
            currentEvent.setNext(nextEvent);
            currentEvent = nextEvent;
            currentEventCount++;
            complexEvents = complexEvents.getNext();
        }
        return currentEvent;
    }

    /**
     * Removes from the underlying collection the last element returned by the
     * iterator (optional operation).  This method can be called only once per
     * call to <tt>next</tt>.  The behavior of an iterator is unspecified if
     * the underlying collection is modified while the iteration is in
     * progress in any way other than by calling this method.
     *
     * @throws UnsupportedOperationException if the <tt>remove</tt>
     *                                       operation is not supported by this Iterator.
     * @throws IllegalStateException         if the <tt>next</tt> method has not
     *                                       yet been called, or the <tt>remove</tt> method has already
     *                                       been called after the last call to the <tt>next</tt>
     *                                       method.
     */
    @Override
    public void remove() {
        if (lastReturned == null) {
            throw new IllegalStateException();
        }
        if (previousToLastReturned != null) {
            previousToLastReturned.setNext(lastReturned.getNext());
        } else {
            first = lastReturned.getNext();
            if (first == null) {
                last = null;
            }
        }
        lastReturned.setNext(null);
        streamEventPool.returnEvents(lastReturned);
        lastReturned = null;
        currentEventCount--;
    }
}
