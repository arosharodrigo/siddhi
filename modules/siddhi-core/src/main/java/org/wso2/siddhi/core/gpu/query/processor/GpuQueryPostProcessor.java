package org.wso2.siddhi.core.gpu.query.processor;

import java.nio.ByteBuffer;

import org.wso2.siddhi.core.gpu.event.stream.converter.ConversionGpuEventChunk;

public abstract class GpuQueryPostProcessor {
    protected ByteBuffer outputEventBuffer;
    protected ConversionGpuEventChunk complexEventChunk;
    
    public GpuQueryPostProcessor(ByteBuffer outputEventBuffer, ConversionGpuEventChunk complexEventChunk) {
        this.outputEventBuffer = outputEventBuffer;
        this.complexEventChunk = complexEventChunk;
    }
    
    public abstract void process(ByteBuffer inputEventBuffer, int eventCount);
}
