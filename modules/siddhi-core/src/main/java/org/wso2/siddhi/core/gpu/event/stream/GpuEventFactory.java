package org.wso2.siddhi.core.gpu.event.stream;

import org.wso2.siddhi.core.event.stream.MetaStreamEvent;

import com.lmax.disruptor.EventFactory;


public class GpuEventFactory implements EventFactory<GpuEvent> {

    private int attributeSize;
    
    public GpuEventFactory(MetaStreamEvent metaStreamEvent) {
        this.attributeSize = metaStreamEvent.getOutputData().size();
    }

    public GpuEvent newInstance() {
        return new GpuEvent(attributeSize);
    }
}
