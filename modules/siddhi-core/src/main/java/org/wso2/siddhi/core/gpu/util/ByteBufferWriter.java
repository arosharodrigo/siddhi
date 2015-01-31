package org.wso2.siddhi.core.gpu.util;

import java.nio.ByteBuffer;

public class ByteBufferWriter {
    private ByteBuffer byteBuffer;
    private int bufferIndex;
    
    public ByteBufferWriter(ByteBuffer byteBuffer) {
        this.byteBuffer = byteBuffer;
        Reset();
    }
    
    public void Reset() {
        bufferIndex = 0;
    }
    
    public ByteBuffer getByteBuffer() {
        return byteBuffer;
    }
    
    public int getBufferIndex() { 
        return bufferIndex;
    }
    
    public void setBufferIndex(int index) {
        bufferIndex = index;
    }

    public void writeByte(byte value) {
        byteBuffer.put(bufferIndex, value);
        bufferIndex += 1;
    }
    
    public void writeChar(char value) {
        byteBuffer.putChar(bufferIndex, value);
        bufferIndex += 2;
    }
    
    public void writeBool(boolean value) {
        byteBuffer.putShort(bufferIndex, (short) (value ? 1 : 0));
        bufferIndex += 2;
    }
    
    public void writeInt(int value) {
        byteBuffer.putInt(bufferIndex, value);
        bufferIndex += 4;
    }
    
    public void writeShort(short value) {
        byteBuffer.putShort(bufferIndex, value);
        bufferIndex += 2;
    }
    
    public void writeLong(long value) {
        byteBuffer.putLong(bufferIndex, value);
        bufferIndex += 8;
    }

    public void writeFloat(float value) {
        byteBuffer.putFloat(bufferIndex, value);
        bufferIndex += 4;
    }

    public void writeDouble(double value) {
        byteBuffer.putDouble(bufferIndex, value);
        bufferIndex += 8;
    }
    
    public void writeString(String value, int length) {
        byte[] str = value.getBytes();
        byteBuffer.putShort((short) str.length);
        bufferIndex += 2;
        byteBuffer.put(str, bufferIndex, str.length);
        bufferIndex += length;
    }
    
}
