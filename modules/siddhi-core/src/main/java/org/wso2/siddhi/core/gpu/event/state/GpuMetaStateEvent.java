package org.wso2.siddhi.core.gpu.event.state;

import org.wso2.siddhi.core.gpu.event.GpuMetaEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.query.api.definition.StreamDefinition;

public class GpuMetaStateEvent implements GpuMetaEvent {
    
    private GpuMetaStreamEvent[] metaStreamEvents;
    private int streamEventCount = 0;
    private StreamDefinition outputStreamDefinition;
    
    public GpuMetaStateEvent(int size) {
        metaStreamEvents = new GpuMetaStreamEvent[size];
    }
    
    public GpuMetaStreamEvent getMetaStreamEvent(int position) {
        if(position >=0 && position < metaStreamEvents.length)
            return metaStreamEvents[position];
        
        return null;
    }
    
    public void addEvent(GpuMetaStreamEvent metaStreamEvent) {
        metaStreamEvents[streamEventCount] = metaStreamEvent;
        streamEventCount++;
    }
    
    public void setOutputDefinition(StreamDefinition streamDefinition) {
        this.outputStreamDefinition = streamDefinition;
    }

    public StreamDefinition getOutputStreamDefinition() {
        return outputStreamDefinition;
    }
    
    public int getStreamCount()
    {
        return streamEventCount;
    }
}
