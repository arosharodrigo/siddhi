package org.wso2.siddhi.core.gpu.query.processor;

import java.nio.ByteBuffer;

import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.converter.ConversionGpuEventChunk;

public class GpuJoinQueryPostProcessor extends GpuQueryPostProcessor {

    public GpuJoinQueryPostProcessor(ByteBuffer outputEventBuffer, ConversionGpuEventChunk complexEventChunks[]) {
        super(outputEventBuffer, complexEventChunks);
    }

    @Override
    public void process(int streamIndex, ByteBuffer inputEventBuffer, int eventCount) {
        outputEventBuffer.position(0);
        inputEventBuffer.position(0);
        
        complexEventChunks[streamIndex].convertAndAdd(outputEventBuffer, eventCount);

    }

}
