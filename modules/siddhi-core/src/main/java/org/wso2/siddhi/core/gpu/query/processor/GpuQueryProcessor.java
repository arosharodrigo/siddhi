package org.wso2.siddhi.core.gpu.query.processor;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.log4j.Logger;
import org.bytedeco.javacpp.BytePointer;
import org.wso2.siddhi.core.event.ComplexEventChunk;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.event.stream.converter.ConversionStreamEventChunk;
import org.wso2.siddhi.core.exception.DefinitionNotExistException;
import org.wso2.siddhi.core.gpu.config.GpuQueryContext;
import org.wso2.siddhi.core.gpu.event.GpuMetaEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.gpu.event.stream.converter.ConversionGpuEventChunk;
import org.wso2.siddhi.core.gpu.util.ByteBufferWriter;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.gpu.jni.SiddhiGpu;

public class GpuQueryProcessor {

    private static final Logger log = Logger.getLogger(GpuQueryProcessor.class);
    private String queryName;
    private Processor selectProcessor;
    private ByteBufferWriter streamInputEventBuffers[];
    private Map<String, GpuMetaStreamEvent> metaStreams = new HashMap<String, GpuMetaStreamEvent>();
    private GpuMetaEvent gpuMetaEvent;
    private ConversionGpuEventChunk complexEventChunks[];
    private final AtomicLong sequenceNumber;
    private SiddhiGpu.GpuQueryRuntime gpuQueryRuntime;
    private GpuQueryContext gpuQueryContext;
    private SiddhiGpu.GpuStreamProcessor gpuStreamProcessor[];
    private List<SiddhiGpu.GpuProcessor> gpuProcessors = new ArrayList<SiddhiGpu.GpuProcessor>();
    private GpuQueryPostProcessor gpuQueryPostProcessor;
    
    public GpuQueryProcessor(GpuMetaEvent gpuMetaEvent, GpuQueryContext gpuQueryContext, String queryName) {
        this.gpuMetaEvent = gpuMetaEvent;
        this.gpuQueryContext = gpuQueryContext;
        this.queryName = queryName;
        this.gpuStreamProcessor = null;
        this.streamInputEventBuffers = null;
        this.gpuQueryPostProcessor = null;
        this.complexEventChunks = null;
        
        sequenceNumber = new AtomicLong(0);
        
        log.info("<" + this.queryName + "> Creating SiddhiGpu.GpuQueryRuntime using device [" + 
                gpuQueryContext.getCudaDeviceId() + "] input buffer size [" + gpuQueryContext.getInputEventBufferSize() + "]");
        
        gpuQueryRuntime = new SiddhiGpu.GpuQueryRuntime(this.queryName, gpuQueryContext.getCudaDeviceId(), 
                gpuQueryContext.getInputEventBufferSize());
    }
    
    public GpuQueryProcessor clone() {
        GpuQueryProcessor clonedQueryProcessor = new GpuQueryProcessor(gpuMetaEvent, gpuQueryContext, this.queryName);
        return clonedQueryProcessor;
    }
    
    public void process(int streamIndex, int eventCount) {
        //log.debug("[process] streamIndex=" + streamIndex + " eventCount=" + eventCount);

        // reset bytebuffer writers
        streamInputEventBuffers[streamIndex].Reset();
        
        // start gpu processor
        int resultEventCount = gpuStreamProcessor[streamIndex].Process(eventCount);
        
        // copy events back
        //log.debug("[process] convert result events : " + resultEventCount);
        
        gpuQueryPostProcessor.process(streamIndex, streamInputEventBuffers[streamIndex].getByteBuffer(), resultEventCount);
        // convert to StreamEvent
        //// streamEventChunk.convertAndAdd(event);
        //// processAndClearGpu(streamEventChunk);
        
        //log.debug("[process] call selector");
        // call Selector processor
        selectProcessor.process(complexEventChunks[streamIndex]);
        complexEventChunks[streamIndex].clear();
        
        //log.debug("[process] complete");
    }
    
    public void setSelectProcessor(Processor selectProcessor) {
        this.selectProcessor = selectProcessor;
    }
    
    public ByteBufferWriter getStreamInputEventBuffer(int streamIndex) {
        return streamInputEventBuffers[streamIndex];
    }
    
    public void addStream(String streamId, GpuMetaStreamEvent metaStreamEvent) {
        
        log.info("<" + queryName + "> [addStream] StreamId=" + streamId + " StreamIndex=" + metaStreamEvent.getStreamIndex() + 
                " AttributeCount=" + metaStreamEvent.getAttributes().size() + 
                " SizeOfEvent=" + metaStreamEvent.getEventSizeInBytes());
        
        metaStreams.put(streamId, metaStreamEvent);

        SiddhiGpu.GpuMetaEvent siddhiGpuMetaEvent = new SiddhiGpu.GpuMetaEvent(metaStreamEvent.getStreamIndex(), 
                metaStreamEvent.getAttributes().size(), metaStreamEvent.getEventSizeInBytes());

        int index = 0;
        for (GpuEventAttribute attrib : metaStreamEvent.getAttributes()) {
            int dataType = -1;

            switch(attrib.type) {
            case BOOL:
                dataType = SiddhiGpu.DataType.Boolean;
                break;
            case DOUBLE:
                dataType = SiddhiGpu.DataType.Double;
                break;
            case FLOAT:
                dataType = SiddhiGpu.DataType.Float;
                break;
            case INT:
                dataType = SiddhiGpu.DataType.Int;
                break;
            case LONG:
                dataType = SiddhiGpu.DataType.Long;
                break;
            case STRING:
                dataType = SiddhiGpu.DataType.StringIn;
                break;
            default:
                break;
            }

            siddhiGpuMetaEvent.SetAttribute(index++, dataType, attrib.length, attrib.position);    
        }

        gpuQueryRuntime.AddStream(metaStreamEvent.getStreamId(), siddhiGpuMetaEvent);
        
    }
    
    public int getStreamIndex(String streamId) {
        GpuMetaStreamEvent metaEvent = metaStreams.get(streamId);
        if(metaEvent != null) {
            return metaEvent.getStreamIndex();
        }
        
        throw new DefinitionNotExistException("StreamId " + streamId + " not found in GpuQueryProcessor");
    }
    
    public long getNextSequenceNumber() {
        return sequenceNumber.getAndIncrement();
    }
    
    public void setComplexEventChunk(int streamIndex, ConversionGpuEventChunk complexEventChunk) {
        this.complexEventChunks[streamIndex] = complexEventChunk;
    }
    
    public SiddhiGpu.GpuQueryRuntime getGpuQueryRuntime() {
        return gpuQueryRuntime;
    }

    public void AddGpuProcessor(SiddhiGpu.GpuProcessor gpuProcessor) {
        log.info("<" + queryName + "> AddGpuProcessor : " + gpuProcessor.GetType());
        gpuProcessors.add(gpuProcessor);
    }
    
    public void configure() {
        
        log.info("<" + queryName + "> configure");
        
        gpuStreamProcessor = new SiddhiGpu.GpuStreamProcessor[metaStreams.size()];
        streamInputEventBuffers = new ByteBufferWriter[metaStreams.size()];
        complexEventChunks = new ConversionGpuEventChunk[metaStreams.size()];
        
        if(gpuQueryRuntime.Configure()) {

            for(Entry<String, GpuMetaStreamEvent> entry : metaStreams.entrySet()) {
                int streamIndex = entry.getValue().getStreamIndex();
                String streamId = entry.getValue().getStreamId();

                gpuStreamProcessor[streamIndex] = gpuQueryRuntime.GetStream(streamId);
                
                BytePointer bytePointer = gpuQueryRuntime.GetInputEventBuffer(new BytePointer(streamId));
                int bufferSize = gpuQueryRuntime.GetInputEventBufferSizeInBytes(streamId);
                bytePointer.capacity(bufferSize);
                bytePointer.limit(bufferSize);
                bytePointer.position(0);
                ByteBuffer eventByteBuffer = bytePointer.asBuffer();
                
                log.info("<" + queryName + "> ByteBuffer for StreamId=" + streamId + " StreamIndex=" + streamIndex + " [" + eventByteBuffer + "]");
                
                streamInputEventBuffers[streamIndex] = new ByteBufferWriter(eventByteBuffer);
            }
            
        } else {
            log.warn("<" + queryName + "> SiddhiGpu.QueryRuntime initialization failed");
            return;
        }
        
        SiddhiGpu.GpuProcessor lastGpuProcessor = gpuProcessors.get(gpuProcessors.size() - 1);
        if(lastGpuProcessor != null) {
            if(lastGpuProcessor instanceof SiddhiGpu.GpuFilterProcessor) {
                
                BytePointer bytePointer = ((SiddhiGpu.GpuFilterProcessor)lastGpuProcessor).GetResultEventBuffer();
                int bufferSize = ((SiddhiGpu.GpuFilterProcessor)lastGpuProcessor).GetResultEventBufferSize();
                bytePointer.capacity(bufferSize);
                bytePointer.limit(bufferSize);
                bytePointer.position(0);
                ByteBuffer eventByteBuffer = bytePointer.asBuffer();
                
                gpuQueryPostProcessor = new GpuFilterQueryPostProcessor(
                        eventByteBuffer,
                        complexEventChunks);
                
            } else if (lastGpuProcessor instanceof SiddhiGpu.GpuLengthSlidingWindowProcessor) {
                
                BytePointer bytePointer = ((SiddhiGpu.GpuLengthSlidingWindowProcessor)lastGpuProcessor).GetResultEventBuffer();
                int bufferSize = ((SiddhiGpu.GpuLengthSlidingWindowProcessor)lastGpuProcessor).GetResultEventBufferSize();
                bytePointer.capacity(bufferSize);
                bytePointer.limit(bufferSize);
                bytePointer.position(0);
                ByteBuffer eventByteBuffer = bytePointer.asBuffer();
                
                gpuQueryPostProcessor = new GpuLengthWindowQueryPostProcessor(
                        eventByteBuffer,
                        complexEventChunks);
                
            } else if(lastGpuProcessor instanceof SiddhiGpu.GpuJoinProcessor) {
                
                BytePointer bytePointer = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetResultEventBuffer();
                int bufferSize = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetResultEventBufferSize();
                bytePointer.capacity(bufferSize);
                bytePointer.limit(bufferSize);
                bytePointer.position(0);
                ByteBuffer eventByteBuffer = bytePointer.asBuffer();
                
                gpuQueryPostProcessor = new GpuJoinQueryPostProcessor(
                        eventByteBuffer,
                        complexEventChunks);
            }
        }
    }

}
