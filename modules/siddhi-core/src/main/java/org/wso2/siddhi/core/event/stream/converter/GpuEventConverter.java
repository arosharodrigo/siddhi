package org.wso2.siddhi.core.event.stream.converter;

import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.event.stream.GpuEvent;
import org.wso2.siddhi.core.event.stream.StreamEvent;

public class GpuEventConverter {

    private void convertToInnerStreamEvent(Object[] data, ComplexEvent.Type type, long timestamp, GpuEvent borrowedEvent) {
        System.arraycopy(data, 0, borrowedEvent.getOutputData(), 0, data.length);
        borrowedEvent.setType(type);
        borrowedEvent.setTimestamp(timestamp);
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
