package org.wso2.siddhi.core.event.stream;

import static org.wso2.siddhi.core.util.SiddhiConstants.STREAM_ATTRIBUTE_INDEX;

import java.util.Arrays;

import org.wso2.siddhi.core.event.ComplexEvent;

public class GpuEvent implements ComplexEvent {

    protected long timestamp = -1;
    protected Object[] attributeData;   
    protected Type type = Type.CURRENT;
    private GpuEvent next;
    
    public GpuEvent(int attributeSize) {
        if(attributeSize > 0) {            
            attributeData = new Object[attributeSize];
        }
    }
    
    @Override
    public ComplexEvent getNext() {
        return next;
    }

    @Override
    public void setNext(ComplexEvent events) {
        this.next = (GpuEvent) events;
    }

    @Override
    public Object[] getOutputData() {
        return attributeData;
    }
    
    public void setOutputData(Object[] outputData) {
        this.attributeData = outputData;
    }

    @Override
    public void setOutputData(Object object, int index) {
        attributeData[index] = object;
    }

    @Override
    public long getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    @Override
    public Object getAttribute(int[] position) {
        return attributeData[position[STREAM_ATTRIBUTE_INDEX]];
    }

    public void setType(Type type) {
        this.type = type;
    }
    
    @Override
    public Type getType() {
        return type;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof GpuEvent)) return false;

        GpuEvent event = (GpuEvent) o;

        if (type != event.type) return false;
        if (timestamp != event.timestamp) return false;
        if (!Arrays.equals(attributeData, event.attributeData)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (timestamp ^ (timestamp >>> 32));
        result = 31 * result + (attributeData != null ? Arrays.hashCode(attributeData) : 0);
        result = 31 * result + type.hashCode();
        return result;
    }

    @Override
    public String toString() {
        final StringBuffer sb = new StringBuffer("GpuEvent{");
        sb.append("timestamp=").append(timestamp);
        sb.append(", attributeData=").append(attributeData == null ? "null" : Arrays.asList(attributeData).toString());
        sb.append(", type=").append(type);
        sb.append(", next=").append(next);
        sb.append('}');
        return sb.toString();
    }
}
