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
import org.wso2.siddhi.core.gpu.query.input.GpuProcessStreamReceiver;
import org.wso2.siddhi.core.gpu.util.ByteBufferWriter;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.gpu.jni.SiddhiGpu;

public class GpuQueryProcessor {

    private static final Logger log = Logger.getLogger(GpuQueryProcessor.class);
    private String queryName;
    private Map<String, GpuMetaStreamEvent> metaStreams = new HashMap<String, GpuMetaStreamEvent>();
    private final AtomicLong sequenceNumber;
    private SiddhiGpu.GpuQueryRuntime gpuQueryRuntime;
    private GpuQueryContext gpuQueryContext;
    private Map<String, GpuProcessStreamReceiver> processStreamReceivers = new HashMap<String, GpuProcessStreamReceiver>();
    
    public GpuQueryProcessor(GpuQueryContext gpuQueryContext, String queryName) {
        this.gpuQueryContext = gpuQueryContext;
        this.queryName = queryName;
        
        sequenceNumber = new AtomicLong(0);
        
        log.info("<" + this.queryName + "> Creating SiddhiGpu.GpuQueryRuntime using device [" + 
                gpuQueryContext.getCudaDeviceId() + "] input buffer size [" + gpuQueryContext.getEventBatchMaximumSize() + "]");
        
        gpuQueryRuntime = new SiddhiGpu.GpuQueryRuntime(this.queryName, gpuQueryContext.getCudaDeviceId(), 
                gpuQueryContext.getEventBatchMaximumSize());
    }
    
    public GpuQueryProcessor clone() {
        GpuQueryProcessor clonedQueryProcessor = new GpuQueryProcessor(gpuQueryContext, this.queryName);
        return clonedQueryProcessor;
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
    
    public SiddhiGpu.GpuQueryRuntime getGpuQueryRuntime() {
        return gpuQueryRuntime;
    }
    
    public GpuQueryContext getGpuQueryContext() {
        return gpuQueryContext;
    }

    public void addGpuProcessor(String streamId, SiddhiGpu.GpuProcessor gpuProcessor) {
        log.info("<" + queryName + "> AddGpuProcessor : ToStream=" + streamId + 
                " Type=" + gpuProcessor.GetType() + " Class=" + gpuProcessor.getClass().getName());
        gpuQueryRuntime.AddProcessor(streamId, gpuProcessor);
    }
    
    public void configure(GpuProcessStreamReceiver processStreamReceiver) {
        
        log.info("<" + queryName + "> configure " + processStreamReceiver.getStreamId() + " : MetaStream count " + metaStreams.size());
        
        processStreamReceivers.put(processStreamReceiver.getStreamId(), processStreamReceiver);
        
        if(processStreamReceivers.size() == metaStreams.size()) {
            log.info("All streams added to GpuQueryRuntime. Configure GpuQueryRuntime.");
            
            if(gpuQueryRuntime.Configure()) {

                for(Entry<String, GpuMetaStreamEvent> entry : metaStreams.entrySet()) {
                    int streamIndex = entry.getValue().getStreamIndex();
                    String myStreamId = entry.getValue().getStreamId();

                    SiddhiGpu.GpuStreamProcessor gpuStreamProcessor = gpuQueryRuntime.GetStream(myStreamId);

                    BytePointer bytePointer = gpuQueryRuntime.GetInputEventBuffer(new BytePointer(myStreamId));
                    int bufferSize = gpuQueryRuntime.GetInputEventBufferSizeInBytes(myStreamId);
                    bytePointer.capacity(bufferSize);
                    bytePointer.limit(bufferSize);
                    bytePointer.position(0);
                    ByteBuffer eventByteBuffer = bytePointer.asBuffer();

                    log.info("<" + queryName + "> ByteBuffer for StreamId=" + myStreamId + " StreamIndex=" + streamIndex + " [" + eventByteBuffer + "]");

                    ByteBufferWriter streamInputEventBuffer = new ByteBufferWriter(eventByteBuffer);
                    
                    processStreamReceivers.get(myStreamId).configure(streamInputEventBuffer, gpuStreamProcessor);
                }

            } else {
                log.warn("<" + queryName + "> SiddhiGpu.QueryRuntime initialization failed");
                return;
            }
        } else {
            log.info("Waiting until all streams add to GpuQueryRuntime");
        }
        
    }

}
