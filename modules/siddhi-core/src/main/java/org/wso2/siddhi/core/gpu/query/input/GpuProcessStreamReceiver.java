package org.wso2.siddhi.core.gpu.query.input;

import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.bytedeco.javacpp.BytePointer;
import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.ComplexEventChunk;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.event.stream.converter.ConversionStreamEventChunk;
import org.wso2.siddhi.core.gpu.event.stream.GpuEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuEventPool;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.gpu.event.stream.converter.ConversionGpuEventChunk;
import org.wso2.siddhi.core.gpu.query.processor.GpuFilterQueryPostProcessor;
import org.wso2.siddhi.core.gpu.query.processor.GpuJoinQueryPostProcessor;
import org.wso2.siddhi.core.gpu.query.processor.GpuLengthWindowQueryPostProcessor;
import org.wso2.siddhi.core.gpu.query.processor.GpuQueryPostProcessor;
import org.wso2.siddhi.core.gpu.query.processor.GpuQueryProcessor;
import org.wso2.siddhi.core.gpu.util.ByteBufferWriter;
import org.wso2.siddhi.core.query.input.ProcessStreamReceiver;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.gpu.jni.SiddhiGpu;
import org.wso2.siddhi.gpu.jni.SiddhiGpu.GpuStreamProcessor;

public class GpuProcessStreamReceiver extends ProcessStreamReceiver {

    private static final Logger log = Logger.getLogger(GpuProcessStreamReceiver.class);
    private GpuQueryProcessor gpuQueryProcessor;
    private GpuMetaStreamEvent gpuMetaEvent;
    private ConversionGpuEventChunk gpuEventChunk;
    private int streamIndex;
    private ByteBufferWriter eventBufferWriter;
    private SiddhiGpu.GpuStreamProcessor gpuStreamProcessor;
    private List<SiddhiGpu.GpuProcessor> gpuProcessors = new ArrayList<SiddhiGpu.GpuProcessor>();
    private GpuQueryPostProcessor gpuQueryPostProcessor;
    private Processor selectProcessor;
    
    private float currentEventCount = 0;
    private long iteration = 0;
    private long startTime = 0;
    private long endTime = 0;
    private long duration = 0;
    private final List<Double> throughputList = new ArrayList<Double>();
    
    private final DecimalFormat decimalFormat = new DecimalFormat("###.##");
    
    public GpuProcessStreamReceiver(String streamId, String queryName) {
        super(streamId, queryName);
        this.gpuQueryProcessor = null;
        this.gpuMetaEvent = null;
        this.gpuEventChunk = null;
        this.eventBufferWriter = null;
        this.currentEventCount = 0;
        this.gpuStreamProcessor = null;
        this.gpuQueryPostProcessor = null;
        this.selectProcessor = null;
    }

    public GpuProcessStreamReceiver clone(String key) {
        GpuProcessStreamReceiver clonedProcessStreamReceiver = new GpuProcessStreamReceiver(streamId + key, queryName);
        clonedProcessStreamReceiver.setMetaStreamEvent(metaStreamEvent);
        clonedProcessStreamReceiver.setGpuQueryProcessor(gpuQueryProcessor.clone());
        return clonedProcessStreamReceiver;
    }
    
    @Override
    public void receive(Event event, boolean endOfBatch) {
        
        log.debug("<" + queryName + " - " + streamId + "> [receive] Event=" + event.toString() + " endOfBatch="+ endOfBatch);
        
        ComplexEvent.Type type = event.isExpired() ? StreamEvent.Type.EXPIRED : StreamEvent.Type.CURRENT;
        eventBufferWriter.writeShort((short)type.ordinal());
        eventBufferWriter.writeLong(gpuQueryProcessor.getNextSequenceNumber());
        eventBufferWriter.writeLong(event.getTimestamp());
        
        Object [] data = event.getData();
        
        int index = 0;
        for (GpuEventAttribute attrib : gpuMetaEvent.getAttributes()) {
            //log.debug("[receive] writing attribute index=" + index + " attrib=" + attrib.toString() + " val=" + data[index] + 
            //        " BufferIndex=" + eventBufferWriter.getBufferIndex() + " BufferPosition=" + eventBufferWriter.getBufferPosition());
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
        currentEventCount++;
        
        if (endOfBatch) {

            startTime = System.nanoTime();
            
            eventBufferWriter.Reset();
  
            int resultEventCount = gpuStreamProcessor.Process((int)currentEventCount);
            
            gpuQueryPostProcessor.process(eventBufferWriter.getByteBuffer(), resultEventCount);
            
            selectProcessor.process(gpuEventChunk);
            gpuEventChunk.clear();
            
            
            endTime = System.nanoTime();
            
            duration = endTime - startTime;
            double average = (currentEventCount * 1000000000 / (double)duration);
            //log.info("Batch Throughput : [" + currentEventCount + "/" + duration + "] " + decimalFormat.format(average) + " eps");
            
            throughputList.add(average);
            
            currentEventCount = 0;
            iteration++;
            
            if(iteration % 100000 == 0)
            {
                double totalThroughput = 0;
                
                for (Double tp : throughputList) {
                    totalThroughput += tp;
                }
                
                double avgThroughput = totalThroughput / throughputList.size();
                log.info("<" + queryName + " - " + streamId + "> Batch Throughput : " + decimalFormat.format(avgThroughput) + " eps");
                throughputList.clear();
            }
        }
    }
  
    public void init() {
        
        log.info("<" + queryName + "> [GpuProcessStreamReceiver] Initializing " + streamId );
        
        for(SiddhiGpu.GpuProcessor gpuProcessor: gpuProcessors) {
            gpuQueryProcessor.addGpuProcessor(streamId, gpuProcessor);
        }

        gpuQueryProcessor.configure(this);     
        
    }
    
    public void configure(ByteBufferWriter eventBufferWriter, GpuStreamProcessor gpuStreamProcessor) {
        streamIndex = gpuQueryProcessor.getStreamIndex(getStreamId());
        log.info("<" + queryName + "> [GpuProcessStreamReceiver] configure : StreamId=" + getStreamId() + " StreamIndex=" + streamIndex);

        this.eventBufferWriter = eventBufferWriter;
        this.gpuStreamProcessor = gpuStreamProcessor;
        
        // create QueryPostProcessor - should done after GpuStreamProcessors configured
        SiddhiGpu.GpuProcessor lastGpuProcessor = gpuProcessors.get(gpuProcessors.size() - 1);
        if(lastGpuProcessor != null) {
            if(lastGpuProcessor instanceof SiddhiGpu.GpuFilterProcessor) {
                
                BytePointer bytePointer = ((SiddhiGpu.GpuFilterProcessor)lastGpuProcessor).GetResultEventBuffer();
                int bufferSize = ((SiddhiGpu.GpuFilterProcessor)lastGpuProcessor).GetResultEventBufferSize();
                bytePointer.capacity(bufferSize);
                bytePointer.limit(bufferSize);
                bytePointer.position(0);
                ByteBuffer eventByteBuffer = bytePointer.asBuffer();
                
                gpuEventChunk = new ConversionGpuEventChunk(metaStreamEvent, streamEventPool, gpuMetaEvent);
                
                gpuQueryPostProcessor = new GpuFilterQueryPostProcessor(
                        eventByteBuffer,
                        gpuEventChunk);
                
            } else if (lastGpuProcessor instanceof SiddhiGpu.GpuLengthSlidingWindowProcessor) {
                
                BytePointer bytePointer = ((SiddhiGpu.GpuLengthSlidingWindowProcessor)lastGpuProcessor).GetResultEventBuffer();
                int bufferSize = ((SiddhiGpu.GpuLengthSlidingWindowProcessor)lastGpuProcessor).GetResultEventBufferSize();
                bytePointer.capacity(bufferSize);
                bytePointer.limit(bufferSize);
                bytePointer.position(0);
                ByteBuffer eventByteBuffer = bytePointer.asBuffer();
                
                gpuEventChunk = new ConversionGpuEventChunk(metaStreamEvent, streamEventPool, gpuMetaEvent);

                gpuQueryPostProcessor = new GpuLengthWindowQueryPostProcessor(
                        eventByteBuffer,
                        gpuEventChunk);
                
            } else if(lastGpuProcessor instanceof SiddhiGpu.GpuJoinProcessor) {
                
                ByteBuffer eventByteBuffer = null;
                int segmentEventCount  = 0;
                
                if(streamIndex == 0) {

                    BytePointer bytePointer = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetLeftResultEventBuffer();
                    int bufferSize = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetLeftResultEventBufferSize();
                    segmentEventCount = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetRightStreamWindowSize();
                    bytePointer.capacity(bufferSize);
                    bytePointer.limit(bufferSize);
                    bytePointer.position(0);
                    eventByteBuffer = bytePointer.asBuffer();

                } else if(streamIndex == 1) {
                    
                    BytePointer bytePointer = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetRightResultEventBuffer();
                    int bufferSize = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetRightResultEventBufferSize();
                    segmentEventCount = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetLeftStreamWindowSize();
                    bytePointer.capacity(bufferSize);
                    bytePointer.limit(bufferSize);
                    bytePointer.position(0);
                    eventByteBuffer = bytePointer.asBuffer();
                    
                }
                
                // TODO: this should be metastream event of output stream
                gpuEventChunk = new ConversionGpuEventChunk(metaStreamEvent, streamEventPool, gpuMetaEvent);
                
                gpuQueryPostProcessor = new GpuJoinQueryPostProcessor(
                        eventByteBuffer,
                        gpuEventChunk,
                        segmentEventCount);
            }
        }
    }

    public void setGpuQueryProcessor(GpuQueryProcessor gpuQueryProcessor) {
        this.gpuQueryProcessor = gpuQueryProcessor;
    }
    
    public GpuQueryProcessor getGpuQueryProcessor() {
        return gpuQueryProcessor;
    }

    public GpuMetaStreamEvent getGpuMetaEvent() {
        return gpuMetaEvent;
    }

    public void setGpuMetaEvent(GpuMetaStreamEvent gpuMetaEvent) {
        this.gpuMetaEvent = gpuMetaEvent;
    }
    
    public void setSelectProcessor(Processor selectProcessor) {
        this.selectProcessor = selectProcessor;
    }
    
    public void addGpuProcessor(SiddhiGpu.GpuProcessor gpuProcessor) {
        log.info("<" + queryName + "> [GpuProcessStreamReceiver] AddGpuProcessor : Type=" + gpuProcessor.GetType() +
                " Class=" + gpuProcessor.getClass().getName());
        gpuProcessors.add(gpuProcessor);
    }
    
    public List<SiddhiGpu.GpuProcessor> getGpuProcessors() {
        return gpuProcessors;
    }
}
