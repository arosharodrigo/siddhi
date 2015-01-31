package org.wso2.siddhi.core.gpu.query.processor;

import java.nio.ByteBuffer;

import org.wso2.siddhi.core.gpu.event.stream.converter.ConversionGpuEventChunk;

public abstract class GpuQueryPostProcessor {
    protected ByteBuffer outputEventBuffer;
    protected ConversionGpuEventChunk complexEventChunks[];
    
    public GpuQueryPostProcessor(ByteBuffer outputEventBuffer, ConversionGpuEventChunk complexEventChunks[]) {
        this.outputEventBuffer = outputEventBuffer;
        this.complexEventChunks = complexEventChunks;
    }
    
    public abstract void process(int streamIndex, ByteBuffer inputEventBuffer, int eventCount);
}
