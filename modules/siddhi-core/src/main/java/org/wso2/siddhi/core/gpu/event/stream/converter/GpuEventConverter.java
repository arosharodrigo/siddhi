package org.wso2.siddhi.core.gpu.event.stream.converter;

import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;

public class GpuEventConverter {

    private GpuMetaStreamEvent metaEvent;
    
    public GpuEventConverter(GpuMetaStreamEvent metaEvent) {
        this.metaEvent = metaEvent;
    }
    
    private void convertToInnerStreamEvent(Object[] data, ComplexEvent.Type type, long timestamp, GpuEvent borrowedEvent) {
        borrowedEvent.setTimestamp(timestamp);
        borrowedEvent.setSequence(1);
        borrowedEvent.setType(type);
        borrowedEvent.setAttributes(data, data.length);
    }
    
    public void convertEvent(Event event, GpuEvent borrowedEvent) {
        convertToInnerStreamEvent(event.getData(), event.isExpired() ? StreamEvent.Type.EXPIRED : StreamEvent.Type.CURRENT,
                event.getTimestamp(), borrowedEvent);

    }

    public void convertStreamEvent(ComplexEvent complexEvent, GpuEvent borrowedEvent) {
        convertToInnerStreamEvent(complexEvent.getOutputData(), complexEvent.getType(),
                complexEvent.getTimestamp(), borrowedEvent);
    }

    public void convertData(long timeStamp, Object[] data, GpuEvent borrowedEvent) {
        convertToInnerStreamEvent(data, StreamEvent.Type.CURRENT, timeStamp, borrowedEvent);

    }

}
