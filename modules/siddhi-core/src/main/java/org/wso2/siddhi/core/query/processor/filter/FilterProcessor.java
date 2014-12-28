/*
 * Copyright (c) 2005 - 2014, WSO2 Inc. (http://www.wso2.org)
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
package org.wso2.siddhi.core.query.processor.filter;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;
import org.wso2.siddhi.core.event.state.MetaStateEvent;
import org.wso2.siddhi.core.event.stream.MetaStreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEventIterator;
import org.wso2.siddhi.core.exception.OperationNotSupportedException;
import org.wso2.siddhi.core.executor.ExpressionExecutor;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.core.util.SiddhiConstants;
import org.wso2.siddhi.gpu.jni.SiddhiGpu;
import org.wso2.siddhi.query.api.definition.Attribute;


public class FilterProcessor implements Processor {

    private static final Logger log = Logger.getLogger(FilterProcessor.class);
    protected Processor next;
    private ExpressionExecutor conditionExecutor;
    private SiddhiGpu.GpuEventConsumer gpuEventConsumer = null;
    private int gpuProcessMinimumEventCount = 256;
    private StreamEvent [] inputStreamEvents = null;
    private Map<Integer, Attribute> varPositionToAttribNameMap = null;
    private int inputStreamEventIndex = 0;
    private int [] stringAttributeSizes = null;
    private String queryName = null;

    private ByteBuffer eventByteBuffer = null;
    private IntBuffer resultsBuffer = null;
    private int filterResultsBufferPosition = 0;
    private int eventsDataBufferPosition = 0;
    private int eventMetaBufferPosition = 0;

    private static class AttributeDefinition {
        public int attributePositionInGpu;
        public int [] attributePositionInCpu;
        public Attribute.Type attributeType;
        public int attributeLength;
    }

    private List<AttributeDefinition> attributeDefinitionList = new ArrayList<AttributeDefinition>();

    public FilterProcessor(ExpressionExecutor conditionExecutor) {
        if(Attribute.Type.BOOL.equals(conditionExecutor.getReturnType())) {
            this.conditionExecutor = conditionExecutor;
        }else{
            throw new OperationNotSupportedException("Return type of "+conditionExecutor.toString()+" should be of type BOOL. " +
                    "Actual type: "+conditionExecutor.getReturnType().toString());
        }
    }

    public FilterProcessor(ExpressionExecutor conditionExecutor, SiddhiGpu.GpuEventConsumer gpuEventConsumer, 
            String queryName, int threshold, String stringAttributeSizes) {
        this.gpuEventConsumer = gpuEventConsumer;
        this.gpuProcessMinimumEventCount = threshold;
        this.queryName = queryName;

        String [] tokens = stringAttributeSizes.split(",");
        if (tokens.length > 0) {
            this.stringAttributeSizes = new int[tokens.length];

            int index = 0;
            for (String string : tokens) {
                this.stringAttributeSizes[index++] = Integer.parseInt(string);
            }
        }

        log.info("[" + this.queryName + "] GpuEventConsumer MaxNumberOfEvents : " + gpuEventConsumer.GetMaxNumberOfEvents());

        this.inputStreamEvents = new StreamEvent[gpuEventConsumer.GetMaxNumberOfEvents()];

        if(Attribute.Type.BOOL.equals(conditionExecutor.getReturnType())) {
            this.conditionExecutor = conditionExecutor;
        }else{
            throw new OperationNotSupportedException("Return type of "+conditionExecutor.toString()+" should be of type BOOL. " +
                    "Actual type: "+conditionExecutor.getReturnType().toString());
        }
    }

    public FilterProcessor cloneProcessor(){
        return new FilterProcessor(conditionExecutor.cloneExecutor());
    }

    public void setVariablePositionToAttributeNameMapper(Map<Integer, Attribute> mapper)
    {
        varPositionToAttribNameMap = mapper;
    }

    @Override
    public void process(StreamEvent event) {

        if (gpuEventConsumer == null) {
            StreamEventIterator iterator = event.getIterator();
            while (iterator.hasNext()) {
                StreamEvent streamEvent = iterator.next();
                if (!(Boolean) conditionExecutor.execute(streamEvent)) {
                    iterator.remove();
                }
            }

            if (iterator.getFirstElement() != null) {
                this.next.process(iterator.getFirstElement());
            }
        } else {

            // check batch size and use GPU processing if size exceed minimum threshold
            // number of events in batch should at least exceed block size

            long preStartTime = System.nanoTime();

            inputStreamEventIndex = 0;
            int bufferIndex = eventsDataBufferPosition; 

            StreamEventIterator iterator = event.getIterator();
            while (iterator.hasNext()) {

                StreamEvent streamEvent = iterator.next();
                log.info("[" + queryName + "] Into GPU " + streamEvent.toString()); 
                inputStreamEvents[inputStreamEventIndex++] = streamEvent;

                for (AttributeDefinition attributeDefinition : attributeDefinitionList) {
                    Object attrib = streamEvent.getAttribute(attributeDefinition.attributePositionInCpu);

                    switch (attributeDefinition.attributeType) {
                    case BOOL: {
                        eventByteBuffer.putShort(bufferIndex, (short) (((Boolean) attrib).booleanValue() ? 1 : 0));
                        bufferIndex += 2;
                    }
                    break;
                    case INT: {
                        eventByteBuffer.putInt(bufferIndex, ((Integer) attrib).intValue()); 
                        bufferIndex += 4;
                    }
                    break;
                    case LONG: {
                        eventByteBuffer.putLong(bufferIndex, ((Long) attrib).longValue());
                        bufferIndex += 8;
                    }
                    break;
                    case FLOAT: {
                        eventByteBuffer.putFloat(bufferIndex, ((Float) attrib).floatValue());
                        bufferIndex += 4;
                    }
                    break;
                    case DOUBLE: {
                        eventByteBuffer.putDouble(bufferIndex, ((Double) attrib).doubleValue());
                        bufferIndex += 8;
                    }
                    break;
                    case STRING: {
                        byte[] str = attrib.toString().getBytes();
                        eventByteBuffer.putShort(bufferIndex, (short) str.length);
                        bufferIndex += 2;
                        eventByteBuffer.put(str, bufferIndex, str.length);
                        bufferIndex += attributeDefinition.attributeLength;
                    }
                    break;
                    default:
                    {
                        log.warn("[" + queryName + "] Unknown attribute [CpuPos=" + attributeDefinition.attributePositionInCpu + 
                                "|Type=" + attributeDefinition.attributeType + "|Attr=" + attrib + "]");
                    }
                    break;
                    }
                }
            }

            if (inputStreamEventIndex >= gpuProcessMinimumEventCount) {

                long preStopTime = System.nanoTime();

                // process events and set results in same buffer
                gpuEventConsumer.ProcessEvents(inputStreamEventIndex);

                long postStartTime = System.nanoTime();

                // read results from byteBuffer
                // max number of result is number of input events to kernel
                resultsBuffer.position(0);

                StreamEvent resultStreamEvent = null;
                StreamEvent lastEvent = null;

                int resultCount = 0;

                for (int resultsIndex = 0; resultsIndex < inputStreamEventIndex; ++resultsIndex) {
                    if (resultsBuffer.get(resultsIndex) == 1) {
                        StreamEvent e = inputStreamEvents[resultsIndex];
                        resultCount++;
                        
                        log.info("[" + queryName + "] Out from GPU " + e.toString());

                        if (lastEvent != null) {
                            lastEvent.setNext(e);
                            lastEvent = e;
                        } else {
                            resultStreamEvent = e;
                            lastEvent = resultStreamEvent;
                        }
                    }
                }

                long postStopTime = System.nanoTime();

                if (resultStreamEvent != null) {
                    log.info("[" + this.queryName + "] InputCount=" + inputStreamEventIndex + " ResultCount=" + resultCount);
                    log.info("[" + this.queryName + "] Times : Pre=" + (preStopTime - preStartTime) + 
                            " Gpu=" + (postStartTime - preStopTime) + 
                            " Post=" + (postStopTime - postStartTime) + 
                            " Total=" + (postStopTime - preStartTime));
                    this.next.process(resultStreamEvent);
                } else {
                    log.info("[" + this.queryName + "] InputCount=" + inputStreamEventIndex + " ResultCount=0");
                    log.info("[" + this.queryName + "] Times : Pre=" + (preStopTime - preStartTime) +
                            " Gpu=" + (postStartTime - preStopTime) + 
                            " Post=" + (postStopTime - postStartTime) + 
                            " Total=" + (postStopTime - preStartTime));
                }

            } else {
                log.info("GPU Threshold not met : [InputCount=" + inputStreamEventIndex + "|Threshold=" + gpuProcessMinimumEventCount + "]");
                
                iterator = event.getIterator();
                while (iterator.hasNext()) {
                    StreamEvent streamEvent = iterator.next();
                    if (!(Boolean) conditionExecutor.execute(streamEvent)) {
                        iterator.remove();
                    }
                }

                if (iterator.getFirstElement() != null) {
                    this.next.process(iterator.getFirstElement());
                }
            }

        }
    }

    @Override
    public Processor getNextProcessor() {
        return next;
    }

    @Override
    public void setNextProcessor(Processor processor) {
        next = processor;
    }

    @Override
    public void setToLast(Processor processor) {
        if (next == null) {
            this.next = processor;
        } else {
            this.next.setNextProcessor(processor);
        }
    }

    @Override
    public void configureProcessor(MetaStateEvent metaEvent) {

        // configure GPU related data structures
        if (metaEvent.getEventCount() == 1 && varPositionToAttribNameMap != null && gpuEventConsumer != null)
        {
            MetaStreamEvent metaStreamEvent = metaEvent.getMetaEvent(0);

            int count = varPositionToAttribNameMap.size();
            int sizeOfEvent = 0;
            int stringAttributeIndex = 0;
            int bufferPreambleSize = 0;

            filterResultsBufferPosition = 0;
            eventMetaBufferPosition = filterResultsBufferPosition + (gpuEventConsumer.GetMaxNumberOfEvents() * 4);

            bufferPreambleSize = eventMetaBufferPosition;
            bufferPreambleSize += 2; // attribute count

            // calculate max byte buffer size
            for(int index = 0; index < count; ++index) {
                Attribute attribute = varPositionToAttribNameMap.get(index);
                if(attribute != null)
                {
                    switch(attribute.getType())
                    {
                    case BOOL:
                    {
                        bufferPreambleSize += 6; // type + length + position
                        sizeOfEvent += 2;
                    }
                    break;
                    case INT:
                    {
                        bufferPreambleSize += 6; // type + length + position
                        sizeOfEvent += 4;
                    }
                    break;
                    case LONG:
                    {
                        bufferPreambleSize += 6; // type + length + position
                        sizeOfEvent += 8;
                    }
                    break;
                    case FLOAT:
                    {
                        bufferPreambleSize += 6; // type + length + position
                        sizeOfEvent += 4;
                    }
                    break;
                    case DOUBLE:
                    {
                        bufferPreambleSize += 6; // type + length + position
                        sizeOfEvent += 8;
                    }
                    break;
                    case STRING:
                    {
                        sizeOfEvent += 2; // actual string length
                        if(stringAttributeSizes != null)
                        {
                            int sizeOfString = stringAttributeSizes[stringAttributeIndex++];
                            sizeOfEvent += sizeOfString; // max size
                        }
                        else
                        {
                            sizeOfEvent += 8;
                        }
                        bufferPreambleSize += 6; // type + length + position
                    }
                    break;
                    default:
                        break;
                    }
                }
            }

            eventsDataBufferPosition = bufferPreambleSize;

            log.info("GpuEventConsumer : Filter results buffer position is " + filterResultsBufferPosition);
            log.info("GpuEventConsumer : EventMeta buffer position is " + eventMetaBufferPosition);
            log.info("GpuEventConsumer : EventData buffer position is " + eventsDataBufferPosition);
            log.info("GpuEventConsumer : Size of an event is " + sizeOfEvent + " bytes");
            int byteBufferSize = eventsDataBufferPosition + (sizeOfEvent * gpuEventConsumer.GetMaxNumberOfEvents());

            gpuEventConsumer.SetSizeOfEvent(sizeOfEvent);
            gpuEventConsumer.SetResultsBufferPosition(filterResultsBufferPosition);
            gpuEventConsumer.SetEventMetaBufferPosition(eventMetaBufferPosition);
            gpuEventConsumer.SetEventDataBufferPosition(eventsDataBufferPosition);

            // allocate byte buffer
            log.info("GpuEventConsumer : Creating ByteBuffer of " + byteBufferSize + " bytes");
            eventByteBuffer = ByteBuffer.allocateDirect(byteBufferSize).order(ByteOrder.nativeOrder());
            resultsBuffer = eventByteBuffer.asIntBuffer();
            log.info("GpuEventConsumer : Created ByteBuffer of " + byteBufferSize + " bytes in [" + eventByteBuffer + "]");
            gpuEventConsumer.SetByteBuffer(eventByteBuffer, byteBufferSize);
            // gpuEventConsumer.CreateByteBuffer(byteBufferSize);
            // eventByteBuffer = gpuEventConsumer.GetByteBuffer().asBuffer();

            log.info("EventByteBuffer : IsDirect=" + this.eventByteBuffer.isDirect() +
                    " HasArray=" + this.eventByteBuffer.hasArray() + 
                    " Position=" + this.eventByteBuffer.position() + 
                    " Limit=" + this.eventByteBuffer.limit());
            

            // fill byte buffer preamble

            int bufferIndex = eventMetaBufferPosition;
            eventByteBuffer.putShort(bufferIndex, (short)count); // put attribute count
            bufferIndex += 2;
            int bufferPosition = 0;

            // fill attribute type - length (2 + 2 bytes)
            for(int index = 0; index < count; ++index) {

                Attribute attribute = varPositionToAttribNameMap.get(index);

                if(attribute != null)
                {
                    AttributeDefinition attributeDefinition = new AttributeDefinition();
                    attributeDefinition.attributePositionInGpu = index;
                    attributeDefinition.attributeType = attribute.getType();

                    if (metaStreamEvent.getOutputData().contains(attribute)) {

                        attributeDefinition.attributePositionInCpu = new int[] {
                                -1, -1,
                                SiddhiConstants.OUTPUT_DATA_INDEX, 
                                metaStreamEvent.getOutputData().indexOf(attribute)};

                    } else if (metaStreamEvent.getAfterWindowData().contains(attribute)) {

                        attributeDefinition.attributePositionInCpu = new int[] {
                                -1, -1,
                                SiddhiConstants.AFTER_WINDOW_DATA_INDEX, 
                                metaStreamEvent.getAfterWindowData().indexOf(attribute)};

                    } else if (metaStreamEvent.getBeforeWindowData().contains(attribute)) {

                        attributeDefinition.attributePositionInCpu = new int[] {
                                -1, -1,
                                SiddhiConstants.BEFORE_WINDOW_DATA_INDEX, 
                                metaStreamEvent.getBeforeWindowData().indexOf(attribute)};
                    } else {

                        continue;
                    }

                    switch(attributeDefinition.attributeType)
                    {
                    case BOOL:
                    {
                        eventByteBuffer.putShort(bufferIndex, (short)SiddhiGpu.DataType.Boolean); // type - 2 bytes
                        bufferIndex += 2;
                        eventByteBuffer.putShort(bufferIndex, (short)2); // length - 2 bytes
                        bufferIndex += 2;
                        eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
                        bufferIndex += 2;
                        bufferPosition += 2;

                        attributeDefinition.attributeLength = 2;
                    }
                    break;
                    case INT:
                    {
                        eventByteBuffer.putShort(bufferIndex, (short)SiddhiGpu.DataType.Int);
                        bufferIndex += 2;
                        eventByteBuffer.putShort(bufferIndex, (short)4); // length - 4 bytes
                        bufferIndex += 2;
                        eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
                        bufferIndex += 2;
                        bufferPosition += 4;

                        attributeDefinition.attributeLength = 4;
                    }
                    break;
                    case LONG:
                    {
                        eventByteBuffer.putShort(bufferIndex, (short)SiddhiGpu.DataType.Long);
                        bufferIndex += 2;
                        eventByteBuffer.putShort(bufferIndex, (short)8); // length - 8 bytes
                        bufferIndex += 2;
                        eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
                        bufferIndex += 2;
                        bufferPosition += 8;							

                        attributeDefinition.attributeLength = 8;
                    }
                    break;
                    case FLOAT:
                    {
                        eventByteBuffer.putShort(bufferIndex, (short)SiddhiGpu.DataType.Float);
                        bufferIndex += 2;
                        eventByteBuffer.putShort(bufferIndex, (short)4); // length - 4 bytes
                        bufferIndex += 2;
                        eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
                        bufferIndex += 2;
                        bufferPosition += 4;

                        attributeDefinition.attributeLength = 4;
                    }
                    break;
                    case DOUBLE:
                    {
                        eventByteBuffer.putShort(bufferIndex, (short)SiddhiGpu.DataType.Double);
                        bufferIndex += 2;
                        eventByteBuffer.putShort(bufferIndex, (short)8); // length - 8 bytes
                        bufferIndex += 2;
                        eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
                        bufferIndex += 2;
                        bufferPosition += 8;

                        attributeDefinition.attributeLength = 8;
                    }
                    break;
                    case STRING:
                    {
                        eventByteBuffer.putShort(bufferIndex, (short)SiddhiGpu.DataType.StringIn);
                        bufferIndex += 2;

                        if(stringAttributeSizes != null)
                        {
                            int sizeOfString = stringAttributeSizes[stringAttributeIndex++];
                            eventByteBuffer.putShort(bufferIndex, (short)sizeOfString); // length - n bytes
                            bufferIndex += 2;
                            eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
                            bufferIndex += 2;
                            bufferPosition += sizeOfString;

                            attributeDefinition.attributeLength = sizeOfString;
                        }
                        else
                        {
                            eventByteBuffer.putShort(bufferIndex, (short)8); // length - 8 bytes
                            bufferIndex += 2;
                            eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
                            bufferIndex += 2;
                            bufferPosition += 8;

                            attributeDefinition.attributeLength = 8;
                        }
                    }
                    break;
                    default:
                        break;
                    }

                    log.info("Attribute : GpuPos=" + attributeDefinition.attributePositionInGpu + 
                            " CpuPos=" + Arrays.toString(attributeDefinition.attributePositionInCpu) + 
                            " Type=" + attributeDefinition.attributeType + 
                            " Length=" + attributeDefinition.attributeLength);
                    attributeDefinitionList.add(attributeDefinition);
                }
            }
        }

        if(this.next != null)
            this.next.configureProcessor(metaEvent);
    }
}
