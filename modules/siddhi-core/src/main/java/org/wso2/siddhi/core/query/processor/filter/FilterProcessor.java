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

import static org.wso2.siddhi.core.util.SiddhiConstants.BEFORE_WINDOW_DATA_INDEX;
import static org.wso2.siddhi.core.util.SiddhiConstants.HAVING_STATE;
import static org.wso2.siddhi.core.util.SiddhiConstants.ON_AFTER_WINDOW_DATA_INDEX;
import static org.wso2.siddhi.core.util.SiddhiConstants.OUTPUT_DATA_INDEX;
import static org.wso2.siddhi.core.util.SiddhiConstants.STATE_OUTPUT_DATA_INDEX;
import static org.wso2.siddhi.core.util.SiddhiConstants.STREAM_ATTRIBUTE_INDEX;
import static org.wso2.siddhi.core.util.SiddhiConstants.STREAM_ATTRIBUTE_TYPE_INDEX;
import static org.wso2.siddhi.core.util.SiddhiConstants.STREAM_EVENT_CHAIN_INDEX;
import static org.wso2.siddhi.core.util.SiddhiConstants.UNKNOWN_STATE;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;
import org.wso2.siddhi.core.event.ComplexEvent;
import org.wso2.siddhi.core.event.ComplexEventChunk;
import org.wso2.siddhi.core.event.MetaComplexEvent;
import org.wso2.siddhi.core.event.state.MetaStateEvent;
import org.wso2.siddhi.core.event.stream.MetaStreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.exception.OperationNotSupportedException;
import org.wso2.siddhi.core.executor.ExpressionExecutor;
import org.wso2.siddhi.core.gpu.util.parser.GpuFilterExpressionParser;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.core.util.SiddhiConstants;
import org.wso2.siddhi.gpu.jni.SiddhiGpu;
import org.wso2.siddhi.query.api.definition.Attribute;


public class FilterProcessor implements Processor {

    private static final Logger log = Logger.getLogger(FilterProcessor.class);
    private static final boolean debugLogEnabled = "true".equals(System.getProperty("debug.log.enable"));
    private static final boolean performanceLogEnabled = "true".equals(System.getProperty("performance.log.enable"));
    
    protected Processor next;
    private ExpressionExecutor conditionExecutor;
    
//    private SiddhiGpu.GpuEventConsumer gpuEventConsumer = null;
//    private int gpuProcessMinimumEventCount = 256;
//    private ComplexEvent [] inputStreamEvents = null;
//    private Map<Integer, GpuFilterExpressionParser.VariablePosition> varPositionToAttribNameMap = null;
//    private int inputStreamEventIndex = 0;
//    private int [] stringAttributeSizes = null;
//    private String queryName = null;
//
//    private ByteBuffer eventByteBuffer = null;
//    private IntBuffer resultsBuffer = null;
//    private int filterResultsBufferPosition = 0;
//    private int eventsDataBufferPosition = 0;
//    private int eventMetaBufferPosition = 0;
//    
//    private long preStartTime = 0, preStopTime = 0, postStartTime = 0, postStopTime = 0;
//
//    private static class AttributeDefinition {
//        public int attributePositionInGpu;
//        public int [] attributePositionInCpu;
//        public Attribute.Type attributeType;
//        public int attributeLength;
//    }
//
//    private List<AttributeDefinition> attributeDefinitionList = new ArrayList<AttributeDefinition>();

    public FilterProcessor(ExpressionExecutor conditionExecutor) {
        if (Attribute.Type.BOOL.equals(conditionExecutor.getReturnType())) {
            this.conditionExecutor = conditionExecutor;
        } else {
            throw new OperationNotSupportedException("Return type of " + conditionExecutor.toString() + " should be of type BOOL. " +
                    "Actual type: " + conditionExecutor.getReturnType().toString());
        }
    }

//    public FilterProcessor(ExpressionExecutor conditionExecutor, SiddhiGpu.GpuEventConsumer gpuEventConsumer, 
//            String queryName, int threshold, String stringAttributeSizes) {
//        this.gpuEventConsumer = gpuEventConsumer;
//        this.gpuProcessMinimumEventCount = threshold;
//        this.queryName = queryName;
//
//        String [] tokens = stringAttributeSizes.split(",");
//        if (tokens.length > 0) {
//            this.stringAttributeSizes = new int[tokens.length];
//
//            int index = 0;
//            for (String string : tokens) {
//                this.stringAttributeSizes[index++] = Integer.parseInt(string);
//            }
//        }
//
//        log.info("[" + this.queryName + "] GpuEventConsumer MaxNumberOfEvents : " + gpuEventConsumer.GetMaxNumberOfEvents());
//
//        this.inputStreamEvents = new ComplexEvent[gpuEventConsumer.GetMaxNumberOfEvents()];
//
//        if(Attribute.Type.BOOL.equals(conditionExecutor.getReturnType())) {
//            this.conditionExecutor = conditionExecutor;
//        }else{
//            throw new OperationNotSupportedException("Return type of "+conditionExecutor.toString()+" should be of type BOOL. " +
//                    "Actual type: "+conditionExecutor.getReturnType().toString());
//        }
//    }
    
    public FilterProcessor cloneProcessor() {
        return new FilterProcessor(conditionExecutor.cloneExecutor());
    }

//    public void setVariablePositionToAttributeNameMapper(Map<Integer, GpuFilterExpressionParser.VariablePosition> mapper)
//    {
//        varPositionToAttribNameMap = mapper;
//    }
    
    @Override
    public void process(ComplexEventChunk complexEventChunk) {
        
//        if (gpuEventConsumer == null) {
            complexEventChunk.reset();
            while (complexEventChunk.hasNext()) {
                ComplexEvent complexEvent = complexEventChunk.next();
                if (!(Boolean) conditionExecutor.execute(complexEvent)) {
                    complexEventChunk.remove();
                }
            }
            if (complexEventChunk.getFirst() != null) {
                this.next.process(complexEventChunk);
            }
//        } else {
//
//            // check batch size and use GPU processing if size exceed minimum threshold
//            // number of events in batch should at least exceed block size
//
//            if (performanceLogEnabled) {
//                preStartTime = System.nanoTime();
//            }
//
//            inputStreamEventIndex = 0;
//            int bufferIndex = eventsDataBufferPosition;
//
//            complexEventChunk.reset();
//            while (complexEventChunk.hasNext()) {
//
//                ComplexEvent complexEvent = complexEventChunk.next();
//                if (debugLogEnabled) {
//                    log.info("[" + queryName + "] Into GPU " + complexEvent.toString());
//                }
//                inputStreamEvents[inputStreamEventIndex++] = complexEvent;
//
//                for (AttributeDefinition attributeDefinition : attributeDefinitionList) {
//                    Object attrib = complexEvent.getAttribute(attributeDefinition.attributePositionInCpu);
//
//                    switch (attributeDefinition.attributeType) {
//                    case BOOL: {
//                        eventByteBuffer.putShort(bufferIndex, (short) (((Boolean) attrib).booleanValue() ? 1 : 0));
//                        bufferIndex += 2;
//                    }
//                        break;
//                    case INT: {
//                        eventByteBuffer.putInt(bufferIndex, ((Integer) attrib).intValue());
//                        bufferIndex += 4;
//                    }
//                        break;
//                    case LONG: {
//                        eventByteBuffer.putLong(bufferIndex, ((Long) attrib).longValue());
//                        bufferIndex += 8;
//                    }
//                        break;
//                    case FLOAT: {
//                        eventByteBuffer.putFloat(bufferIndex, ((Float) attrib).floatValue());
//                        bufferIndex += 4;
//                    }
//                        break;
//                    case DOUBLE: {
//                        eventByteBuffer.putDouble(bufferIndex, ((Double) attrib).doubleValue());
//                        bufferIndex += 8;
//                    }
//                        break;
//                    case STRING: {
//                        byte[] str = attrib.toString().getBytes();
//                        eventByteBuffer.putShort(bufferIndex, (short) str.length);
//                        bufferIndex += 2;
//                        eventByteBuffer.put(str, bufferIndex, str.length);
//                        bufferIndex += attributeDefinition.attributeLength;
//                    }
//                        break;
//                    default: {
//                        log.warn("[" + queryName + "] Unknown attribute [CpuPos=" + attributeDefinition.attributePositionInCpu +
//                                "|Type=" + attributeDefinition.attributeType + "|Attr=" + attrib + "]");
//                    }
//                        break;
//                    }
//                }
//            }
//
//            if (inputStreamEventIndex >= gpuProcessMinimumEventCount) {
//
//                if (performanceLogEnabled) {
//                    preStopTime = System.nanoTime();
//                }
//
//                // process events and set results in same buffer
//                gpuEventConsumer.ProcessEvents(inputStreamEventIndex);
//
//                if (performanceLogEnabled) {
//                    postStartTime = System.nanoTime();
//                }
//
//                // read results from byteBuffer
//                // max number of result is number of input events to kernel
//                resultsBuffer.position(0);
//
//                ComplexEvent resultStreamEvent = null;
//                ComplexEvent lastEvent = null;
//
//                int resultCount = 0;
//
//                for (int resultsIndex = 0; resultsIndex < inputStreamEventIndex; ++resultsIndex) {
//                    int matched = resultsBuffer.get();
//                    if (matched >= 0) {
//                        ComplexEvent e = inputStreamEvents[resultsIndex];
//                        resultCount++;
//
//                        if (debugLogEnabled) {
//                            log.info("[" + matched + "] Out from GPU [" + resultsIndex + "]" + e.toString());
//                        }
//
//                        if (lastEvent != null) {
//                            lastEvent.setNext(e);
//                            lastEvent = e;
//                        } else {
//                            resultStreamEvent = e;
//                            lastEvent = resultStreamEvent;
//                        }
//                    } else if (debugLogEnabled) {
//                        ComplexEvent e = inputStreamEvents[resultsIndex];
//                        log.info("[" + matched + "] Not matched from GPU [" + resultsIndex + "]" + e.toString());
//                    }
//                }
//
//                if (performanceLogEnabled) {
//                    postStopTime = System.nanoTime();
//                }
//
//                if (resultStreamEvent != null) {
//                    if (performanceLogEnabled) {
//                        log.info("[" + this.queryName + "] InputCount=" + inputStreamEventIndex + " ResultCount=" + resultCount);
//                        log.info("[" + this.queryName + "] Times : Pre=" + (preStopTime - preStartTime) +
//                                " Gpu=" + (postStartTime - preStopTime) +
//                                " Post=" + (postStopTime - postStartTime) +
//                                " Total=" + (postStopTime - preStartTime));
//                    }
//
//                    this.next.process(complexEventChunk);
//
//                } else if (performanceLogEnabled) {
//                    log.info("[" + this.queryName + "] InputCount=" + inputStreamEventIndex + " ResultCount=0");
//                    log.info("[" + this.queryName + "] Times : Pre=" + (preStopTime - preStartTime) +
//                            " Gpu=" + (postStartTime - preStopTime) +
//                            " Post=" + (postStopTime - postStartTime) +
//                            " Total=" + (postStopTime - preStartTime));
//                }
//
//            } else {
//                if (debugLogEnabled) {
//                    log.info("GPU Threshold not met : [InputCount=" + inputStreamEventIndex + "|Threshold=" + gpuProcessMinimumEventCount + "]");
//                }
//
//                while (complexEventChunk.hasNext()) {
//                    ComplexEvent complexEvent = complexEventChunk.next();
//                    if (!(Boolean) conditionExecutor.execute(complexEvent)) {
//                        complexEventChunk.remove();
//                    }
//                }
//                if (complexEventChunk.getFirst() != null) {
//                    this.next.process(complexEventChunk);
//                }
//            }
//
//        }
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
            this.next.setToLast(processor);
        }
    }

    @Override
    public void configureProcessor(MetaComplexEvent metaComplexEvent) {
        
        // configure GPU related data structures
//        if (varPositionToAttribNameMap != null && gpuEventConsumer != null)
//        {
//            int count = varPositionToAttribNameMap.size();
//            int sizeOfEvent = 0;
//            int stringAttributeIndex = 0;
//            int bufferPreambleSize = 0;
//
//            filterResultsBufferPosition = 0;
//            eventMetaBufferPosition = filterResultsBufferPosition + (gpuEventConsumer.GetMaxNumberOfEvents() * 4);
//
//            bufferPreambleSize = eventMetaBufferPosition;
//            bufferPreambleSize += 2; // attribute count
//
//            // calculate max byte buffer size
//            for(int index = 0; index < count; ++index) {
//                GpuFilterExpressionParser.VariablePosition varPos = varPositionToAttribNameMap.get(index);
//                if(varPos != null)
//                {
//                    switch(varPos.type)
//                    {
//                    case BOOL:
//                    {
//                        bufferPreambleSize += 6; // type + length + position
//                        sizeOfEvent += 2;
//                    }
//                    break;
//                    case INT:
//                    {
//                        bufferPreambleSize += 6; // type + length + position
//                        sizeOfEvent += 4;
//                    }
//                    break;
//                    case LONG:
//                    {
//                        bufferPreambleSize += 6; // type + length + position
//                        sizeOfEvent += 8;
//                    }
//                    break;
//                    case FLOAT:
//                    {
//                        bufferPreambleSize += 6; // type + length + position
//                        sizeOfEvent += 4;
//                    }
//                    break;
//                    case DOUBLE:
//                    {
//                        bufferPreambleSize += 6; // type + length + position
//                        sizeOfEvent += 8;
//                    }
//                    break;
//                    case STRING:
//                    {
//                        sizeOfEvent += 2; // actual string length
//                        if(stringAttributeSizes != null)
//                        {
//                            int sizeOfString = stringAttributeSizes[stringAttributeIndex++];
//                            sizeOfEvent += sizeOfString; // max size
//                        }
//                        else
//                        {
//                            sizeOfEvent += 8;
//                        }
//                        bufferPreambleSize += 6; // type + length + position
//                    }
//                    break;
//                    default:
//                        break;
//                    }
//                }
//            }
//
//            eventsDataBufferPosition = bufferPreambleSize;
//
//            log.info("[" + queryName + "] GpuEventConsumer : Filter results buffer position is " + filterResultsBufferPosition);
//            log.info("[" + queryName + "] GpuEventConsumer : EventMeta buffer position is " + eventMetaBufferPosition);
//            log.info("[" + queryName + "] GpuEventConsumer : EventData buffer position is " + eventsDataBufferPosition);
//            log.info("[" + queryName + "] GpuEventConsumer : Size of an event is " + sizeOfEvent + " bytes");
//            int byteBufferSize = eventsDataBufferPosition + (sizeOfEvent * gpuEventConsumer.GetMaxNumberOfEvents());
//
//            // Should set these before buffer allocation
//            gpuEventConsumer.SetSizeOfEvent(sizeOfEvent);
//            gpuEventConsumer.SetResultsBufferPosition(filterResultsBufferPosition);
//            gpuEventConsumer.SetEventMetaBufferPosition(eventMetaBufferPosition);
//            gpuEventConsumer.SetEventDataBufferPosition(eventsDataBufferPosition);
//
//            // allocate byte buffer
//            log.info("GpuEventConsumer : Creating ByteBuffer of " + byteBufferSize + " bytes");
//            eventByteBuffer = ByteBuffer.allocateDirect(byteBufferSize).order(ByteOrder.nativeOrder());
//            resultsBuffer = eventByteBuffer.asIntBuffer();
//            log.info("GpuEventConsumer : Created ByteBuffer of " + byteBufferSize + " bytes in [" + eventByteBuffer + "]");
//            gpuEventConsumer.SetByteBuffer(eventByteBuffer, byteBufferSize);
//            // gpuEventConsumer.CreateByteBuffer(byteBufferSize);
//            // eventByteBuffer = gpuEventConsumer.GetByteBuffer().asBuffer();
//
//            log.info("[" + queryName + "] EventByteBuffer : IsDirect=" + this.eventByteBuffer.isDirect() +
//                    " HasArray=" + this.eventByteBuffer.hasArray() + 
//                    " Position=" + this.eventByteBuffer.position() + 
//                    " Limit=" + this.eventByteBuffer.limit());
//            
//
//            // fill byte buffer preamble
//
//            int bufferIndex = eventMetaBufferPosition;
//            eventByteBuffer.putShort(bufferIndex, (short)count); // put attribute count
//            bufferIndex += 2;
//            int bufferPosition = 0;
//            
//            
//
//            // fill attribute type - length (2 + 2 bytes)
//            for(int index = 0; index < count; ++index) {
//
//                GpuFilterExpressionParser.VariablePosition varPos = varPositionToAttribNameMap.get(index);
//
//                if(varPos != null)
//                {
//                    
//                    int streamEventChainIndex = varPos.position[STREAM_EVENT_CHAIN_INDEX];
//                    Attribute attr = new Attribute(varPos.attributeName, varPos.type);
//
//                    if (streamEventChainIndex == HAVING_STATE) {
//                        if (metaComplexEvent instanceof MetaStreamEvent) {
//                            varPos.position[STREAM_ATTRIBUTE_TYPE_INDEX] = OUTPUT_DATA_INDEX;
//                        } else {
//                            varPos.position[STREAM_ATTRIBUTE_TYPE_INDEX] = STATE_OUTPUT_DATA_INDEX;
//                        }
//                        varPos.position[STREAM_EVENT_CHAIN_INDEX] = UNKNOWN_STATE;
//                        varPos.position[STREAM_ATTRIBUTE_INDEX] = 
//                                metaComplexEvent.getOutputStreamDefinition().getAttributeList().indexOf(attr);
//                    } else {
//                        MetaStreamEvent metaStreamEvent;
//                        if (metaComplexEvent instanceof MetaStreamEvent) {
//                            metaStreamEvent = (MetaStreamEvent) metaComplexEvent;
//                        } else {
//                            metaStreamEvent = ((MetaStateEvent) metaComplexEvent).getMetaStreamEvent(streamEventChainIndex);
//                        }
//
//                        if (metaStreamEvent.getOutputData().contains(attr)) {
//                            varPos.position[STREAM_ATTRIBUTE_TYPE_INDEX] = OUTPUT_DATA_INDEX;
//                            varPos.position[STREAM_ATTRIBUTE_INDEX] = metaStreamEvent.getOutputData().indexOf(attr);
//                        } else if (metaStreamEvent.getOnAfterWindowData().contains(attr)) {
//                            varPos.position[STREAM_ATTRIBUTE_TYPE_INDEX] = ON_AFTER_WINDOW_DATA_INDEX;
//                            varPos.position[STREAM_ATTRIBUTE_INDEX] = metaStreamEvent.getOnAfterWindowData().indexOf(attr);
//                        } else if (metaStreamEvent.getBeforeWindowData().contains(attr)) {
//                            varPos.position[STREAM_ATTRIBUTE_TYPE_INDEX] = BEFORE_WINDOW_DATA_INDEX;
//                            varPos.position[STREAM_ATTRIBUTE_INDEX] = metaStreamEvent.getBeforeWindowData().indexOf(attr);
//                        }
//                    }
//                    
//                    AttributeDefinition attributeDefinition = new AttributeDefinition();
//                    attributeDefinition.attributePositionInGpu = index;
//                    attributeDefinition.attributeType = varPos.type;
//                    attributeDefinition.attributePositionInCpu = varPos.position.clone();
//
//                    switch(attributeDefinition.attributeType)
//                    {
//                    case BOOL:
//                    {
//                        eventByteBuffer.putShort(bufferIndex, (short)SiddhiGpu.DataType.Boolean); // type - 2 bytes
//                        bufferIndex += 2;
//                        eventByteBuffer.putShort(bufferIndex, (short)2); // length - 2 bytes
//                        bufferIndex += 2;
//                        eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
//                        bufferIndex += 2;
//                        bufferPosition += 2;
//
//                        attributeDefinition.attributeLength = 2;
//                    }
//                    break;
//                    case INT:
//                    {
//                        eventByteBuffer.putShort(bufferIndex, (short)SiddhiGpu.DataType.Int);
//                        bufferIndex += 2;
//                        eventByteBuffer.putShort(bufferIndex, (short)4); // length - 4 bytes
//                        bufferIndex += 2;
//                        eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
//                        bufferIndex += 2;
//                        bufferPosition += 4;
//
//                        attributeDefinition.attributeLength = 4;
//                    }
//                    break;
//                    case LONG:
//                    {
//                        eventByteBuffer.putShort(bufferIndex, (short)SiddhiGpu.DataType.Long);
//                        bufferIndex += 2;
//                        eventByteBuffer.putShort(bufferIndex, (short)8); // length - 8 bytes
//                        bufferIndex += 2;
//                        eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
//                        bufferIndex += 2;
//                        bufferPosition += 8;                                                    
//
//                        attributeDefinition.attributeLength = 8;
//                    }
//                    break;
//                    case FLOAT:
//                    {
//                        eventByteBuffer.putShort(bufferIndex, (short)SiddhiGpu.DataType.Float);
//                        bufferIndex += 2;
//                        eventByteBuffer.putShort(bufferIndex, (short)4); // length - 4 bytes
//                        bufferIndex += 2;
//                        eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
//                        bufferIndex += 2;
//                        bufferPosition += 4;
//
//                        attributeDefinition.attributeLength = 4;
//                    }
//                    break;
//                    case DOUBLE:
//                    {
//                        eventByteBuffer.putShort(bufferIndex, (short)SiddhiGpu.DataType.Double);
//                        bufferIndex += 2;
//                        eventByteBuffer.putShort(bufferIndex, (short)8); // length - 8 bytes
//                        bufferIndex += 2;
//                        eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
//                        bufferIndex += 2;
//                        bufferPosition += 8;
//
//                        attributeDefinition.attributeLength = 8;
//                    }
//                    break;
//                    case STRING:
//                    {
//                        eventByteBuffer.putShort(bufferIndex, (short)SiddhiGpu.DataType.StringIn);
//                        bufferIndex += 2;
//
//                        if(stringAttributeSizes != null)
//                        {
//                            int sizeOfString = stringAttributeSizes[stringAttributeIndex++];
//                            eventByteBuffer.putShort(bufferIndex, (short)sizeOfString); // length - n bytes
//                            bufferIndex += 2;
//                            eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
//                            bufferIndex += 2;
//                            bufferPosition += sizeOfString;
//
//                            attributeDefinition.attributeLength = sizeOfString;
//                        }
//                        else
//                        {
//                            eventByteBuffer.putShort(bufferIndex, (short)8); // length - 8 bytes
//                            bufferIndex += 2;
//                            eventByteBuffer.putShort(bufferIndex, (short)bufferPosition); // position - 2 bytes
//                            bufferIndex += 2;
//                            bufferPosition += 8;
//
//                            attributeDefinition.attributeLength = 8;
//                        }
//                    }
//                    break;
//                    default:
//                        break;
//                    }
//
//                    log.info("[" + queryName + "] Attribute : GpuPos=" + attributeDefinition.attributePositionInGpu + 
//                            " CpuPos=" + Arrays.toString(attributeDefinition.attributePositionInCpu) + 
//                            " Type=" + attributeDefinition.attributeType + 
//                            " Length=" + attributeDefinition.attributeLength);
//                    attributeDefinitionList.add(attributeDefinition);
//                }
//            }
//        }

        if(this.next != null)
            this.next.configureProcessor(metaComplexEvent);
    }
    
}
