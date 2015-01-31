package org.wso2.siddhi.core.gpu.event.stream;

import org.wso2.siddhi.core.gpu.util.ByteBufferWriter;

public class GpuEventPool {
    private GpuEventFactory eventFactory;
    private GpuMetaStreamEvent metaEvent;
    private ByteBufferWriter bufferWriter;
    private int size;
    private int index = 0;
    private GpuEvent streamEventList;

    public GpuEventPool(GpuMetaStreamEvent gpuMetaEvent, ByteBufferWriter bufferWriter, int size) {
        eventFactory = new GpuEventFactory(gpuMetaEvent, bufferWriter);
        this.metaEvent = gpuMetaEvent;
        this.bufferWriter = bufferWriter;
        this.size = size;
    }

    /**
     * Borrowing an StreamEvent
     *
     * @return if StreamEvent exist in the pool an existing event if not a new StreamEvent will be returned
     */
    public GpuEvent borrowEvent() {
        if (index > 0) {
            GpuEvent event = streamEventList;
            streamEventList = (GpuEvent) streamEventList.getNext();
            event.setNext(null);
            index--;
            return event;
        } else {
            return eventFactory.newInstance();
        }
    }

    /**
     * Collects the used InnerStreamEvents
     * If the pool has space the returned event will be added to the pool else it will be dropped
     *
     * @param gpuEvent used event
     */
    public void returnEvents(GpuEvent gpuEvent) {
        if (gpuEvent != null) {
            if (index < size) {
                GpuEvent first = gpuEvent;
                GpuEvent last = gpuEvent;
                while (gpuEvent != null) {
                    last = gpuEvent;
                    index++;
                    gpuEvent = (GpuEvent) gpuEvent.getNext();
                }
                last.setNext(streamEventList);
                streamEventList = first;
            }
        }

    }

    /**
     * @return Occupied buffer size
     */
    public int getBufferedEventsSize() {
        return index;
    }

    public int getSize() {
        return size;
    }
    
    public GpuMetaStreamEvent getGpuMetaEvent() {
        return metaEvent;
    }
    
    public ByteBufferWriter getByteBufferWriter() {
        return bufferWriter;
    }
}
