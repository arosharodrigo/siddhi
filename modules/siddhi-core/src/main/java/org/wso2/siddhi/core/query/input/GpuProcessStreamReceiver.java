package org.wso2.siddhi.core.query.input;

import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.ComplexEventChunk;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.event.stream.GpuEvent;
import org.wso2.siddhi.core.event.stream.GpuEventPool;
import org.wso2.siddhi.core.event.stream.converter.ConversionGpuEventChunk;

public class GpuProcessStreamReceiver extends ProcessStreamReceiver {

    private ConversionGpuEventChunk streamEventChunk;
    private GpuEventPool gpuEventPool;
    
    public GpuProcessStreamReceiver(String streamId) {
        super(streamId);
    }

    public GpuProcessStreamReceiver clone(String key) {
        GpuProcessStreamReceiver clonedProcessStreamReceiver = new GpuProcessStreamReceiver(streamId + key);
        clonedProcessStreamReceiver.setMetaStreamEvent(metaStreamEvent);
        clonedProcessStreamReceiver.setGpuEventPool(new GpuEventPool(metaStreamEvent, gpuEventPool.getSize()));
        return clonedProcessStreamReceiver;
    }
    
    @Override
    public void receive(ComplexEvent complexEvent) {
        streamEventChunk.convertAndAssign(complexEvent);
        processAndClearGpu(streamEventChunk);
    }

    @Override
    public void receive(Event event) {
        streamEventChunk.convertAndAssign(event);
        processAndClearGpu(streamEventChunk);
    }

    @Override
    public void receive(Event[] events) {
        streamEventChunk.convertAndAssign(events);
        processAndClearGpu(streamEventChunk);
    }


    @Override
    public void receive(Event event, boolean endOfBatch) {
        streamEventChunk.convertAndAdd(event);
        if (endOfBatch) {
            processAndClearGpu(streamEventChunk);
        }
    }

    @Override
    public void receive(long timeStamp, Object[] data) {
        streamEventChunk.convertAndAssign(timeStamp, data);
        processAndClearGpu(streamEventChunk);
    }
    
    protected void processAndClearGpu(ComplexEventChunk<GpuEvent> streamEventChunk) {
        if (stateProcessorsSize != 0) {
            stateProcessors.get(0).updateState();
        }
        // If GPU process? call GpuEventProcessor
        
        // Else call next.process(streamEventChunk);
        next.process(streamEventChunk);
        streamEventChunk.clear();
    }
    
    public void setGpuEventPool(GpuEventPool streamEventPool) {
        this.gpuEventPool = streamEventPool;
    }

    public void init() {
        streamEventChunk = new ConversionGpuEventChunk(metaStreamEvent, gpuEventPool);
    }

}
