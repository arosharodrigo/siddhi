/*
 * Copyright (c) 2014, WSO2 Inc. (http://www.wso2.org)
 * All Rights Reserved.
 *
 * WSO2 Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.wso2.siddhi.core.query.input;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;
import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.ComplexEventChunk;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.event.stream.MetaStreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEventPool;
import org.wso2.siddhi.core.event.stream.converter.ConversionStreamEventChunk;
import org.wso2.siddhi.core.query.input.stream.state.PreStateProcessor;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.core.stream.StreamJunction;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class ProcessStreamReceiver implements StreamJunction.Receiver {

    private static final Logger log = Logger.getLogger(ProcessStreamReceiver.class);
    
    protected String streamId;
    protected String queryName;
    protected Processor next;
    protected MetaStreamEvent metaStreamEvent;
    protected ConversionStreamEventChunk streamEventChunk;
    protected StreamEventPool streamEventPool;
    protected List<PreStateProcessor> stateProcessors = new ArrayList<PreStateProcessor>();
    protected int stateProcessorsSize;
    
    private float currentEventCount = 0;
    private long iteration = 0;
    private long startTime = 0;
    private long endTime = 0;
    private long duration = 0;
    private long totalDuration = 0;
    private final SummaryStatistics throughputStatstics = new SummaryStatistics();
    protected int perfromanceCalculateBatchCount;

    private final DecimalFormat decimalFormat = new DecimalFormat("###.##");

    public ProcessStreamReceiver(String streamId) {
        this.streamId = streamId;
        this.queryName = streamId;
        this.perfromanceCalculateBatchCount = 1000;
    }
    
    public ProcessStreamReceiver(String streamId, String queryName) {
        this.streamId = streamId;
        this.queryName = queryName;
        this.perfromanceCalculateBatchCount = 1000;
    }

    @Override
    public String getStreamId() {
        return streamId;
    }

    public ProcessStreamReceiver clone(String key) {
        ProcessStreamReceiver clonedProcessStreamReceiver = new ProcessStreamReceiver(streamId + key, queryName);
        clonedProcessStreamReceiver.setMetaStreamEvent(metaStreamEvent);
        clonedProcessStreamReceiver.setStreamEventPool(new StreamEventPool(metaStreamEvent, streamEventPool.getSize()));
        return clonedProcessStreamReceiver;
    }

    @Override
    public void receive(ComplexEvent complexEvent) {
        startTime = System.nanoTime();
        
        streamEventChunk.convertAndAssign(complexEvent);
        processAndClear(streamEventChunk);
        
        endTime = System.nanoTime();
        totalDuration += (endTime - startTime);
        currentEventCount++;
        
        if(currentEventCount == perfromanceCalculateBatchCount) {
           
            double avgThroughput = currentEventCount * 1000000000 / totalDuration;
            log.info("<" + queryName + "> " + perfromanceCalculateBatchCount + " Events Throughput : " + decimalFormat.format(avgThroughput) + " eps");
            
            throughputStatstics.addValue(avgThroughput);
            totalDuration = 0;
            currentEventCount = 0;
        }
    }

    @Override
    public void receive(Event event) {
        startTime = System.nanoTime();
        
        streamEventChunk.convertAndAssign(event);
        processAndClear(streamEventChunk);
        
        endTime = System.nanoTime();
        totalDuration += (endTime - startTime);
        currentEventCount++;
        
        if(currentEventCount == perfromanceCalculateBatchCount) {
          
            double avgThroughput = currentEventCount * 1000000000 / totalDuration;
            log.info("<" + queryName + "> " + perfromanceCalculateBatchCount + " Events Throughput : " + decimalFormat.format(avgThroughput) + " eps");
            
            throughputStatstics.addValue(avgThroughput);
            
            totalDuration = 0;
            currentEventCount = 0;
        }
    }

    @Override
    public void receive(Event[] events) {
        startTime = System.nanoTime();
        
        streamEventChunk.convertAndAssign(events);
        processAndClear(streamEventChunk);
        
        endTime = System.nanoTime();
        totalDuration += (endTime - startTime);
        currentEventCount += events.length;
        
        if(currentEventCount >= perfromanceCalculateBatchCount) {
            double avgThroughput = currentEventCount * 1000000000 / totalDuration;
            log.info("<" + queryName + "> " + currentEventCount + " Events Throughput : " + decimalFormat.format(avgThroughput) + " eps");
            
            throughputStatstics.addValue(avgThroughput);
            
            totalDuration = 0;
            currentEventCount = 0;
        }
    }


    @Override
    public void receive(Event event, boolean endOfBatch) {
        streamEventChunk.convertAndAdd(event);
        currentEventCount++;
        if (endOfBatch) {
            startTime = System.nanoTime();
            
            processAndClear(streamEventChunk);
            
            endTime = System.nanoTime();
            
            duration = endTime - startTime;
            double average = (currentEventCount * 1000000000 / (double)duration);
            //log.info("<" + streamId + "> Batch Throughput : [" + currentEventCount + "/" + duration + "] " + decimalFormat.format(average) + " eps");
            throughputStatstics.addValue(average);
            
            currentEventCount = 0;
//            iteration++;
//            
//            if(iteration % 100000 == 0)
//            {
//                double totalThroughput = 0;
//                
//                for (Double tp : throughputList) {
//                    totalThroughput += tp;
//                }
//                
//                double avgThroughput = totalThroughput / throughputList.size();
//                log.info("<" + queryName + "> Batch Throughput : " + decimalFormat.format(avgThroughput) + " eps");
//                throughputList.clear();
//            }
        }
    }

    @Override
    public void receive(long timeStamp, Object[] data) {
        startTime = System.nanoTime();
        
        streamEventChunk.convertAndAssign(timeStamp, data);
        processAndClear(streamEventChunk);
        
        endTime = System.nanoTime();
        totalDuration += (endTime - startTime);
        currentEventCount++;
        
        if(currentEventCount == perfromanceCalculateBatchCount) {
           
            double avgThroughput = currentEventCount * 1000000000 / totalDuration;
            log.info("<" + queryName + "> " + perfromanceCalculateBatchCount + " Events Throughput : " + decimalFormat.format(avgThroughput) + " eps");
            
            throughputStatstics.addValue(avgThroughput);
            
            totalDuration = 0;
            currentEventCount = 0;
        }
    }
    
    protected void processAndClear(ComplexEventChunk<StreamEvent> streamEventChunk) {
        //System.out.println("ProcessStreamReceiver [" + this + " / " + Thread.currentThread().getName() + "] " + streamEventChunk.getCurrentEventCount());
        if (stateProcessorsSize != 0) {
            stateProcessors.get(0).updateState();
        }
        next.process(streamEventChunk);
        streamEventChunk.clear();
    }

    public void setMetaStreamEvent(MetaStreamEvent metaStreamEvent) {
        this.metaStreamEvent = metaStreamEvent;
    }

    public MetaStreamEvent getMetaStreamEvent() {
        return metaStreamEvent;
    }

    public void setNext(Processor next) {
        this.next = next;
    }

    public void setStreamEventPool(StreamEventPool streamEventPool) {
        this.streamEventPool = streamEventPool;
    }

    public void init() {
        streamEventChunk = new ConversionStreamEventChunk(metaStreamEvent, streamEventPool);
    }

    public void addStatefulProcessor(PreStateProcessor stateProcessor) {
        stateProcessors.add(stateProcessor);
        stateProcessorsSize = stateProcessors.size();
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
    }
    
    public void getStatistics(List<SummaryStatistics> statList) {
        statList.add(throughputStatstics);
    }
    
    public int getPerfromanceCalculateBatchCount() {
        return perfromanceCalculateBatchCount;
    }

    public void setPerfromanceCalculateBatchCount(int perfromanceCalculateBatchCount) {
        this.perfromanceCalculateBatchCount = perfromanceCalculateBatchCount;
    }
}
