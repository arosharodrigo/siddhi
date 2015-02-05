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
    
    public int getBufferPosition() {
        return byteBuffer.position();
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
        byteBuffer.putShort(bufferIndex, (short) str.length);
        bufferIndex += 2;
        byteBuffer.position(bufferIndex);
        byteBuffer.put(str, 0, str.length); // watch out: This can go wrong if length < str.length
        bufferIndex += length;
    }
    
}
