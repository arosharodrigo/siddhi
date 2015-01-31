package org.wso2.siddhi.core.gpu.event.stream;

import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.gpu.util.ByteBufferWriter;

public class GpuEvent implements ComplexEvent {
    private long timestamp; //TODO: remove these variables to reduce event size
    private Type type;
    private GpuEvent next;
    private GpuMetaStreamEvent metaEvent;
    private ByteBufferWriter eventBufferWriter;
    int eventStartPosition;
    
    public GpuEvent(GpuMetaStreamEvent metaEvent, ByteBufferWriter bufferWriter) {
        this.metaEvent = metaEvent;
        this.eventBufferWriter = bufferWriter;
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
        return null;
    }
    
    public void setOutputData(Object[] outputData) {
    }

    @Override
    public void setOutputData(Object object, int index) {
    }

    @Override
    public long getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
        eventStartPosition = eventBufferWriter.getBufferIndex();
        eventBufferWriter.writeLong(timestamp);
    }

    public void setSequence(long sequence) {
        eventBufferWriter.writeLong(sequence);
    }
    
    @Override
    public Object getAttribute(int[] position) {
        return null;
    }
    
    public void setAttributes(Object [] data, int length) {
        int index = 0;
        for (GpuEventAttribute attrib : metaEvent.getAttributes()) {
            switch(attrib.type) {
            case BOOL:
                eventBufferWriter.writeBool(((Boolean) data[index++]).booleanValue());
                break;
            case INT:
                eventBufferWriter.writeInt(((Integer) data[index++]).intValue());
                break;
            case LONG:
                eventBufferWriter.writeLong(((Long) data[index++]).longValue());
                break;
            case FLOAT:
                eventBufferWriter.writeFloat(((Float) data[index++]).floatValue());
                break;
            case DOUBLE:
                eventBufferWriter.writeDouble(((Double) data[index++]).doubleValue());
                break;
            case STRING: 
                eventBufferWriter.writeString((String) data[index++], attrib.length);
                break;
            }
        }
    }

    public void setType(Type type) {
        this.type = type;
        eventBufferWriter.writeByte((byte) type.ordinal());
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

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (timestamp ^ (timestamp >>> 32));
        result = 31 * result + type.hashCode();
        return result;
    }

    @Override
    public String toString() {
        final StringBuffer sb = new StringBuffer("GpuEvent{");
        sb.append("timestamp=").append(timestamp);
        sb.append(", type=").append(type);
        sb.append(", next=").append(next);
        sb.append('}');
        return sb.toString();
    }
}
