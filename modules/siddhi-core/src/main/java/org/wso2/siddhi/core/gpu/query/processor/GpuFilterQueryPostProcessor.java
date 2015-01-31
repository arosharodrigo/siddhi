package org.wso2.siddhi.core.gpu.query.processor;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.converter.ConversionGpuEventChunk;

public class GpuFilterQueryPostProcessor extends GpuQueryPostProcessor {
    private IntBuffer outputEventIndexBuffer;
    
    public GpuFilterQueryPostProcessor(ByteBuffer outputEventBuffer, ConversionGpuEventChunk complexEventChunks[]) {
        super(outputEventBuffer, complexEventChunks);
        outputEventIndexBuffer = outputEventBuffer.asIntBuffer();
    }

    @Override
    public void process(int streamIndex, ByteBuffer inputEventBuffer, int eventCount) {
        outputEventIndexBuffer.position(0);
        inputEventBuffer.position(0);
        
        complexEventChunks[streamIndex].convertAndAdd(outputEventIndexBuffer, inputEventBuffer, eventCount);
               
    }

}
