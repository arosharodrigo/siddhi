package org.wso2.siddhi.core.gpu.query.input;

import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;
import org.bytedeco.javacpp.BytePointer;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent;
import org.wso2.siddhi.core.gpu.event.stream.GpuMetaStreamEvent.GpuEventAttribute;
import org.wso2.siddhi.core.gpu.query.processor.GpuQueryProcessor;
import org.wso2.siddhi.core.gpu.query.selector.GpuJoinQuerySelector;
import org.wso2.siddhi.core.gpu.query.selector.GpuQuerySelector;
import org.wso2.siddhi.core.gpu.util.ByteBufferWriter;
import org.wso2.siddhi.core.gpu.util.parser.GpuInputStreamParser;
import org.wso2.siddhi.core.query.input.ProcessStreamReceiver;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.gpu.jni.SiddhiGpu;
import org.wso2.siddhi.gpu.jni.SiddhiGpu.GpuStreamProcessor;
import org.wso2.siddhi.query.api.definition.StreamDefinition;

public class GpuProcessStreamReceiver extends ProcessStreamReceiver {

    private static final Logger log = Logger.getLogger(GpuProcessStreamReceiver.class);
    private GpuQueryProcessor gpuQueryProcessor;
    private GpuMetaStreamEvent gpuMetaEvent;
    private int streamIndex;
    private ByteBufferWriter eventBufferWriter;
    private SiddhiGpu.GpuStreamProcessor gpuStreamProcessor;
    private List<SiddhiGpu.GpuProcessor> gpuProcessors = new ArrayList<SiddhiGpu.GpuProcessor>();
    private GpuQuerySelector selectProcessor;
    private int maximumEventBatchSize;
    private int minimumEventBatchSize;
    private boolean softBatchScheduling;
    
    private float currentEventCount = 0;
    private long iteration = 0;
    private long startTime = 0;
    private long endTime = 0;
    private long gpuProcEndTime = 0;
    private long serializeBeginTime = 0;
    private long serializeTime = 0;
    private long duration = 0;
    private final SummaryStatistics throughputStatstics = new SummaryStatistics();
    private final SummaryStatistics totalTimeStatstics = new SummaryStatistics();
    private final SummaryStatistics serializationTimeStatstics = new SummaryStatistics();
    private final SummaryStatistics gpuTimeStatstics = new SummaryStatistics();
    private final SummaryStatistics selectTimeStatstics = new SummaryStatistics();
    private int perfromanceCalculateBatchCount;
    
    private final DecimalFormat decimalFormat = new DecimalFormat("###.##");
    
    public GpuProcessStreamReceiver(String streamId, String queryName) {
        super(streamId, queryName);
        this.gpuQueryProcessor = null;
        this.gpuMetaEvent = null;
        this.eventBufferWriter = null;
        this.currentEventCount = 0;
        this.gpuStreamProcessor = null;
        this.selectProcessor = null;
        this.maximumEventBatchSize = 1024;
        this.minimumEventBatchSize = 1;
        this.softBatchScheduling = true;
        this.perfromanceCalculateBatchCount = 1000;
    }

    public GpuProcessStreamReceiver clone(String key) {
        GpuProcessStreamReceiver clonedProcessStreamReceiver = GpuInputStreamParser.getGpuProcessStreamReceiver(
                this.gpuMetaEvent, streamId + key, queryName);
        
        if(clonedProcessStreamReceiver == null) {
            clonedProcessStreamReceiver = new GpuProcessStreamReceiver(streamId + key, queryName);
        }
        
        clonedProcessStreamReceiver.setMetaStreamEvent(metaStreamEvent);
        clonedProcessStreamReceiver.setGpuQueryProcessor(gpuQueryProcessor.clone());
        clonedProcessStreamReceiver.setGpuMetaEvent(this.gpuMetaEvent);
        clonedProcessStreamReceiver.setPerfromanceCalculateBatchCount(this.perfromanceCalculateBatchCount);
        clonedProcessStreamReceiver.setSoftBatchScheduling(this.softBatchScheduling);
        clonedProcessStreamReceiver.setMaximumEventBatchSize(this.maximumEventBatchSize);
        clonedProcessStreamReceiver.setMinimumEventBatchSize(this.minimumEventBatchSize);
        
        return clonedProcessStreamReceiver;
    }
    
    public void serialize(Event event) {
        
        eventBufferWriter.writeShort((short)(!event.isExpired() ? 0 : 1));
        eventBufferWriter.writeLong(gpuQueryProcessor.getNextSequenceNumber());
        eventBufferWriter.writeLong(event.getTimestamp());

        Object [] data = event.getData();

        int index = 0;
        for (GpuEventAttribute attrib : gpuMetaEvent.getAttributes()) {
            //          log.debug("[receive] writing attribute index=" + index + " attrib=" + attrib.toString() + " val=" + data[index] + 
            //                  " BufferIndex=" + eventBufferWriter.getBufferIndex() + " BufferPosition=" + eventBufferWriter.getBufferPosition());
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

//        // ----- hardcoded logic -----------
//        //        sid string
//        eventBufferWriter.writeString((String) data[index++], 8);
//        //        ts long
//        eventBufferWriter.writeLong(((Long) data[index++]).longValue());
//        //        x int
//        eventBufferWriter.writeInt(((Integer) data[index++]).intValue());
//        //        y int
//        eventBufferWriter.writeInt(((Integer) data[index++]).intValue());
//        //        z int
//        eventBufferWriter.writeInt(((Integer) data[index++]).intValue());
//        //        v double
//        eventBufferWriter.writeDouble(((Double) data[index++]).doubleValue());
//        //        a double
//        eventBufferWriter.writeDouble(((Double) data[index++]).doubleValue());
//        //        vx int
//        eventBufferWriter.writeInt(((Integer) data[index++]).intValue());
//        //        vy int
//        eventBufferWriter.writeInt(((Integer) data[index++]).intValue());
//        //        vz int
//        eventBufferWriter.writeInt(((Integer) data[index++]).intValue());
//        //        ax int
//        eventBufferWriter.writeInt(((Integer) data[index++]).intValue());
//        //        ay int
//        eventBufferWriter.writeInt(((Integer) data[index++]).intValue());
//        //        az int
//        eventBufferWriter.writeInt(((Integer) data[index++]).intValue());
//        //        tsr long
//        eventBufferWriter.writeLong(((Long) data[index++]).longValue());
//        //        tsms long 
//        eventBufferWriter.writeLong(((Long) data[index++]).longValue());
//        // ---------------------------------
    }
    
    @Override
    public void receive(Event event, boolean endOfBatch) {
        
        //log.debug("<" + queryName + " - " + streamId + "> [receive] Event=" + event.toString() + " endOfBatch="+ endOfBatch);
        
        serializeBeginTime = System.nanoTime();
        
        serialize(event);
        
        serializeTime += (System.nanoTime() - serializeBeginTime);
        
        currentEventCount++;
        
        if ((endOfBatch && (currentEventCount >= minimumEventBatchSize)) || (maximumEventBatchSize == currentEventCount)) { //TODO: implement soft/hard batch scheduling

            startTime = System.nanoTime();
            
            eventBufferWriter.Reset();
  
            int resultEventCount = gpuStreamProcessor.Process((int)currentEventCount);
            
            gpuProcEndTime = System.nanoTime();
            
            selectProcessor.process(resultEventCount);
            
            endTime = System.nanoTime();
            
            duration = endTime - startTime; // + serializeTime
            double average = (currentEventCount * 1000000000 / (double)duration);
            
//            log.debug("<" + queryName + " - " + streamId + "> Batch Times : " + currentEventCount + " [Total=" + (endTime - startTime) + 
//                    " Serialize=" + serializeTime +
//                    " Gpu=" + (gpuProcEndTime - startTime) + 
//                    " Select=" + (endTime - gpuProcEndTime) + "] iter=" + iteration);
            
            throughputStatstics.addValue(average);
            totalTimeStatstics.addValue(endTime - startTime);
            serializationTimeStatstics.addValue(serializeTime);
            gpuTimeStatstics.addValue(gpuProcEndTime - startTime);
            selectTimeStatstics.addValue(endTime - gpuProcEndTime);
            
            currentEventCount = 0;
            serializeTime = 0;
//            iteration++;
            
//            if(iteration % perfromanceCalculateBatchCount == 0)
//            {
//                double totalThroughput = 0;
//                
//                for (Double tp : throughputList) {
//                    totalThroughput += tp;
//                }
//                
//                double avgThroughput = totalThroughput / throughputList.size();
//                log.info("<" + queryName + " - " + streamId + "> Batch Throughput : " + decimalFormat.format(avgThroughput) + " eps");
//                throughputList.clear();
//            }
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
        log.info("<" + queryName + "> [GpuProcessStreamReceiver] configure : metaStreamEvent : " + metaStreamEvent.toString());

        this.eventBufferWriter = eventBufferWriter;
        this.gpuStreamProcessor = gpuStreamProcessor;
        
//        configureSelectorProcessor();
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
        this.selectProcessor = (GpuQuerySelector) selectProcessor;
        
        configureSelectorProcessor();
    }
    
    public void addGpuProcessor(SiddhiGpu.GpuProcessor gpuProcessor) {
        log.info("<" + queryName + "> [GpuProcessStreamReceiver] AddGpuProcessor : Type=" + gpuProcessor.GetType() +
                " Class=" + gpuProcessor.getClass().getName());
        gpuProcessors.add(gpuProcessor);
    }
    
    public List<SiddhiGpu.GpuProcessor> getGpuProcessors() {
        return gpuProcessors;
    }
    
    public int getPerfromanceCalculateBatchCount() {
        return perfromanceCalculateBatchCount;
    }

    public void setPerfromanceCalculateBatchCount(int perfromanceCalculateBatchCount) {
        this.perfromanceCalculateBatchCount = perfromanceCalculateBatchCount;
    }
    
    public int getMaximumEventBatchSize() {
        return maximumEventBatchSize;
    }

    public void setMaximumEventBatchSize(int eventBatchSize) {
        this.maximumEventBatchSize = eventBatchSize;
    }
    
    public void setSoftBatchScheduling(boolean softBatchScheduling) {
        this.softBatchScheduling = softBatchScheduling;
    }

    public int getMinimumEventBatchSize() {
        return minimumEventBatchSize;
    }

    public void setMinimumEventBatchSize(int minimumEventBatchSize) {
        this.minimumEventBatchSize = minimumEventBatchSize;
    }
    
    private void configureSelectorProcessor() {
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

                StreamDefinition outputStreamDef = (StreamDefinition) metaStreamEvent.getInputDefinition();
                GpuMetaStreamEvent outputGpuMetaEvent = new GpuMetaStreamEvent(outputStreamDef.getId(), outputStreamDef, 
                        gpuQueryProcessor.getGpuQueryContext());
                outputGpuMetaEvent.setStreamIndex(0);

                selectProcessor.setInputEventBuffer(eventBufferWriter.getByteBuffer());
                selectProcessor.setOutputEventBuffer(eventByteBuffer);
                selectProcessor.setStreamEventPool(streamEventPool);
                selectProcessor.setMetaStreamEvent(metaStreamEvent);
                selectProcessor.setGpuMetaStreamEvent(outputGpuMetaEvent);
                selectProcessor.setWorkerSize(0);

            } else if (lastGpuProcessor instanceof SiddhiGpu.GpuLengthSlidingWindowProcessor) {

                BytePointer bytePointer = ((SiddhiGpu.GpuLengthSlidingWindowProcessor)lastGpuProcessor).GetResultEventBuffer();
                int bufferSize = ((SiddhiGpu.GpuLengthSlidingWindowProcessor)lastGpuProcessor).GetResultEventBufferSize();
                bytePointer.capacity(bufferSize);
                bytePointer.limit(bufferSize);
                bytePointer.position(0);
                ByteBuffer eventByteBuffer = bytePointer.asBuffer();

                StreamDefinition outputStreamDef = (StreamDefinition) metaStreamEvent.getInputDefinition();
                GpuMetaStreamEvent outputGpuMetaEvent = new GpuMetaStreamEvent(outputStreamDef.getId(), outputStreamDef, 
                        gpuQueryProcessor.getGpuQueryContext());
                outputGpuMetaEvent.setStreamIndex(0);

                selectProcessor.setInputEventBuffer(eventBufferWriter.getByteBuffer());
                selectProcessor.setOutputEventBuffer(eventByteBuffer);
                selectProcessor.setStreamEventPool(streamEventPool);
                selectProcessor.setMetaStreamEvent(metaStreamEvent);
                selectProcessor.setGpuMetaStreamEvent(outputGpuMetaEvent);
                selectProcessor.setWorkerSize(5);

            } else if(lastGpuProcessor instanceof SiddhiGpu.GpuJoinProcessor) {

                ByteBuffer eventByteBuffer = null;
                int segmentEventCount  = 0;
                int threadWorkSize = 0;
                int bufferSize = 0;

                if(streamIndex == 0) {

                    BytePointer bytePointer = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetLeftResultEventBuffer();
                    bufferSize = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetLeftResultEventBufferSize();
                    segmentEventCount = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetRightStreamWindowSize();
                    bytePointer.capacity(bufferSize);
                    bytePointer.limit(bufferSize);
                    bytePointer.position(0);
                    eventByteBuffer = bytePointer.asBuffer();
                    
                    threadWorkSize = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetThreadWorkSize();

                } else if(streamIndex == 1) {

                    BytePointer bytePointer = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetRightResultEventBuffer();
                    bufferSize = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetRightResultEventBufferSize();
                    segmentEventCount = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetLeftStreamWindowSize();
                    bytePointer.capacity(bufferSize);
                    bytePointer.limit(bufferSize);
                    bytePointer.position(0);
                    eventByteBuffer = bytePointer.asBuffer();

                    threadWorkSize = ((SiddhiGpu.GpuJoinProcessor)lastGpuProcessor).GetThreadWorkSize();
                }

                if(threadWorkSize >= segmentEventCount) {
                    threadWorkSize = segmentEventCount;
                }
                
                StreamDefinition outputStreamDef = (StreamDefinition) metaStreamEvent.getInputDefinition();
                GpuMetaStreamEvent outputGpuMetaEvent = new GpuMetaStreamEvent(outputStreamDef.getId(), outputStreamDef, 
                        gpuQueryProcessor.getGpuQueryContext());
                outputGpuMetaEvent.setStreamIndex(0);

                GpuJoinQuerySelector gpuJoinQuerySelector = (GpuJoinQuerySelector) selectProcessor;
                gpuJoinQuerySelector.setInputEventBuffer(eventBufferWriter.getByteBuffer());
                gpuJoinQuerySelector.setOutputEventBuffer(eventByteBuffer);
                gpuJoinQuerySelector.setSegmentEventCount(segmentEventCount);
                gpuJoinQuerySelector.setStreamEventPool(streamEventPool);
                gpuJoinQuerySelector.setMetaStreamEvent(metaStreamEvent);
                gpuJoinQuerySelector.setGpuMetaStreamEvent(outputGpuMetaEvent);
                gpuJoinQuerySelector.setThreadWorkSize(threadWorkSize);
                gpuJoinQuerySelector.setWorkerSize(7); // + 1 selector thread = 8 //TODO: set this from query
                
                int numerOfEventsInOutBuffer = bufferSize / outputGpuMetaEvent.getEventSizeInBytes();
                int segmentCount = numerOfEventsInOutBuffer / segmentEventCount;
                gpuJoinQuerySelector.setSegmentsPerWorker(segmentCount / 8);
            }
        }
    }
    
    public void printStatistics() {
        log.info(new StringBuilder()
        .append("EventProcessTroughput ExecutionPlan=").append(queryName).append("_").append(streamId)
        .append("|length=").append(throughputStatstics.getN())
        .append("|Avg=").append(decimalFormat.format(throughputStatstics.getMean()))
        .append("|Min=").append(decimalFormat.format(throughputStatstics.getMin()))
        .append("|Max=").append(decimalFormat.format(throughputStatstics.getMax()))
        .append("|Var=").append(decimalFormat.format(throughputStatstics.getVariance()))
        .append("|StdDev=").append(decimalFormat.format(throughputStatstics.getStandardDeviation())).toString());
//        .append("|10=").append(decimalFormat.format(throughputStatstics.getPercentile(10)))
//        .append("|90=").append(decimalFormat.format(throughputStatstics.getPercentile(90))).toString());
        
        log.info(new StringBuilder()
        .append("EventProcessTotalTime ExecutionPlan=").append(queryName).append("_").append(streamId)
        .append("|length=").append(totalTimeStatstics.getN())
        .append("|Avg=").append(decimalFormat.format(totalTimeStatstics.getMean()))
        .append("|Min=").append(decimalFormat.format(totalTimeStatstics.getMin()))
        .append("|Max=").append(decimalFormat.format(totalTimeStatstics.getMax()))
        .append("|Var=").append(decimalFormat.format(totalTimeStatstics.getVariance()))
        .append("|StdDev=").append(decimalFormat.format(totalTimeStatstics.getStandardDeviation())).toString());
//        .append("|10=").append(decimalFormat.format(totalTimeStatstics.getPercentile(10)))
//        .append("|90=").append(decimalFormat.format(totalTimeStatstics.getPercentile(90))).toString());
        
        log.info(new StringBuilder()
        .append("EventProcessSerializationTime ExecutionPlan=").append(queryName).append("_").append(streamId)
        .append("|length=").append(serializationTimeStatstics.getN())
        .append("|Avg=").append(decimalFormat.format(serializationTimeStatstics.getMean()))
        .append("|Min=").append(decimalFormat.format(serializationTimeStatstics.getMin()))
        .append("|Max=").append(decimalFormat.format(serializationTimeStatstics.getMax()))
        .append("|Var=").append(decimalFormat.format(serializationTimeStatstics.getVariance()))
        .append("|StdDev=").append(decimalFormat.format(serializationTimeStatstics.getStandardDeviation())).toString());
//        .append("|10=").append(decimalFormat.format(serializationTimeStatstics.getPercentile(10)))
//        .append("|90=").append(decimalFormat.format(serializationTimeStatstics.getPercentile(90))).toString());
        
        log.info(new StringBuilder()
        .append("EventProcessGpuTime ExecutionPlan=").append(queryName).append("_").append(streamId)
        .append("|length=").append(gpuTimeStatstics.getN())
        .append("|Avg=").append(decimalFormat.format(gpuTimeStatstics.getMean()))
        .append("|Min=").append(decimalFormat.format(gpuTimeStatstics.getMin()))
        .append("|Max=").append(decimalFormat.format(gpuTimeStatstics.getMax()))
        .append("|Var=").append(decimalFormat.format(gpuTimeStatstics.getVariance()))
        .append("|StdDev=").append(decimalFormat.format(gpuTimeStatstics.getStandardDeviation())).toString());
//        .append("|10=").append(decimalFormat.format(gpuTimeStatstics.getPercentile(10)))
//        .append("|90=").append(decimalFormat.format(gpuTimeStatstics.getPercentile(90))).toString());
        
        log.info(new StringBuilder()
        .append("EventProcessSelectTime ExecutionPlan=").append(queryName).append("_").append(streamId)
        .append("|length=").append(selectTimeStatstics.getN())
        .append("|Avg=").append(decimalFormat.format(selectTimeStatstics.getMean()))
        .append("|Min=").append(decimalFormat.format(selectTimeStatstics.getMin()))
        .append("|Max=").append(decimalFormat.format(selectTimeStatstics.getMax()))
        .append("|Var=").append(decimalFormat.format(selectTimeStatstics.getVariance()))
        .append("|StdDev=").append(decimalFormat.format(selectTimeStatstics.getStandardDeviation())).toString());
//        .append("|10=").append(decimalFormat.format(selectTimeStatstics.getPercentile(10)))
//        .append("|90=").append(decimalFormat.format(selectTimeStatstics.getPercentile(90))).toString());
        
    }
    
    public void getStatistics(List<SummaryStatistics> statList) {
        statList.add(throughputStatstics);
    }
}
