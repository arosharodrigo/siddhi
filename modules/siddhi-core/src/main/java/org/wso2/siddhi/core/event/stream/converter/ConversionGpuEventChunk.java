package org.wso2.siddhi.core.event.stream.converter;

import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.ComplexEventChunk;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.event.stream.GpuEvent;
import org.wso2.siddhi.core.event.stream.GpuEventPool;
import org.wso2.siddhi.core.event.stream.MetaStreamEvent;

public class ConversionGpuEventChunk extends ComplexEventChunk<GpuEvent> {
    private GpuEventConverter gpuEventConverter;
    private GpuEventPool gpuEventPool; 

    public ConversionGpuEventChunk(MetaStreamEvent metaStreamEvent, GpuEventPool gpuEventPool) {
        this.gpuEventPool = gpuEventPool;
        gpuEventConverter = new GpuEventConverter();
    }

    public ConversionGpuEventChunk(GpuEventConverter gpuEventConverter, GpuEventPool gpuEventPool) {
        this.gpuEventConverter = gpuEventConverter;
        this.gpuEventPool = gpuEventPool;
    }
    
    public void convertAndAssign(Event event) {
        GpuEvent borrowedEvent = gpuEventPool.borrowEvent();
        gpuEventConverter.convertEvent(event, borrowedEvent);
        first = borrowedEvent;
        last = first;
        currentEventCount = 1;
    }

    public void convertAndAssign(long timeStamp, Object[] data) {
        GpuEvent borrowedEvent = gpuEventPool.borrowEvent();
        gpuEventConverter.convertData(timeStamp, data, borrowedEvent);
        first = borrowedEvent;
        last = first;
        currentEventCount = 1;
    }

    public void convertAndAssign(ComplexEvent complexEvent) {
        first = gpuEventPool.borrowEvent();
        currentEventCount = 1;
        last = convertAllStreamEvents(complexEvent, first);
    }

//    @Override
//    public void convertAndAssignFirst(StreamEvent streamEvent) {
//        StreamEvent borrowedEvent = streamEventPool.borrowEvent();
//        eventConverter.convertStreamEvent(streamEvent, borrowedEvent);
//        first = borrowedEvent;
//        last = first;
//    }

    public void convertAndAssign(Event[] events) {
        GpuEvent firstEvent = gpuEventPool.borrowEvent();
        gpuEventConverter.convertEvent(events[0], firstEvent);
        GpuEvent currentEvent = firstEvent;
        for (int i = 1, eventsLength = events.length; i < eventsLength; i++) {
            GpuEvent nextEvent = gpuEventPool.borrowEvent();
            gpuEventConverter.convertEvent(events[i], nextEvent);
            currentEvent.setNext(nextEvent);
            currentEvent = nextEvent;
        }
        first = firstEvent;
        last = currentEvent;
        currentEventCount = events.length;
    }

    public void convertAndAdd(Event event) {
        GpuEvent borrowedEvent = gpuEventPool.borrowEvent();
        gpuEventConverter.convertEvent(event, borrowedEvent);

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

    private GpuEvent convertAllStreamEvents(ComplexEvent complexEvents, GpuEvent firstEvent) {
        gpuEventConverter.convertStreamEvent(complexEvents, firstEvent);
        GpuEvent currentEvent = firstEvent;
        complexEvents = complexEvents.getNext();
        while (complexEvents != null) {
            GpuEvent nextEvent = gpuEventPool.borrowEvent();
            gpuEventConverter.convertStreamEvent(complexEvents, nextEvent);
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
            first = (GpuEvent) lastReturned.getNext();
            if (first == null) {
                last = null;
            }
        }
        lastReturned.setNext(null);
        gpuEventPool.returnEvents(lastReturned);
        lastReturned = null;
        currentEventCount--;
    }
}
