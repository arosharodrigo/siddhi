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
    protected Processor next;
    protected MetaStreamEvent metaStreamEvent;
    protected ConversionStreamEventChunk streamEventChunk;
    protected StreamEventPool streamEventPool;
    protected List<PreStateProcessor> stateProcessors = new ArrayList<PreStateProcessor>();
    protected int stateProcessorsSize;
    
    private float currentEventCount = 0;
    private long startTime = 0;
    private long endTime = 0;
    private long duration = 0;

    private final DecimalFormat decimalFormat = new DecimalFormat("###.##");

    public ProcessStreamReceiver(String streamId) {
        this.streamId = streamId;
    }

    @Override
    public String getStreamId() {
        return streamId;
    }

    public ProcessStreamReceiver clone(String key) {
        ProcessStreamReceiver clonedProcessStreamReceiver = new ProcessStreamReceiver(streamId + key);
        clonedProcessStreamReceiver.setMetaStreamEvent(metaStreamEvent);
        clonedProcessStreamReceiver.setStreamEventPool(new StreamEventPool(metaStreamEvent, streamEventPool.getSize()));
        return clonedProcessStreamReceiver;
    }

    @Override
    public void receive(ComplexEvent complexEvent) {
        streamEventChunk.convertAndAssign(complexEvent);
        processAndClear(streamEventChunk);
    }

    @Override
    public void receive(Event event) {
        streamEventChunk.convertAndAssign(event);
        processAndClear(streamEventChunk);
    }

    @Override
    public void receive(Event[] events) {
        streamEventChunk.convertAndAssign(events);
        processAndClear(streamEventChunk);
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
            log.info("Batch Throughput : [" + currentEventCount + "/" + duration + "] " + decimalFormat.format(average) + " eps");
            
            currentEventCount = 0;
        }
    }

    @Override
    public void receive(long timeStamp, Object[] data) {
        streamEventChunk.convertAndAssign(timeStamp, data);
        processAndClear(streamEventChunk);
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
}
