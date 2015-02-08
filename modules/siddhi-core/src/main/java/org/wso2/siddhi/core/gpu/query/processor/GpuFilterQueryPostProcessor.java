package org.wso2.siddhi.core.gpu.query.processor;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.converter.ConversionGpuEventChunk;

public class GpuFilterQueryPostProcessor extends GpuQueryPostProcessor {
    private IntBuffer outputEventIndexBuffer;
    
    public GpuFilterQueryPostProcessor(ByteBuffer outputEventBuffer, ConversionGpuEventChunk complexEventChunk) {
        super(outputEventBuffer, complexEventChunk);
        outputEventIndexBuffer = outputEventBuffer.asIntBuffer();
    }

    @Override
    public void process(ByteBuffer inputEventBuffer, int eventCount) {
        outputEventIndexBuffer.position(0);
        inputEventBuffer.position(0);
        
        complexEventChunk.convertAndAdd(outputEventIndexBuffer, inputEventBuffer, eventCount);
               
    }

}
